import torch
import torch.nn as nn

class SignalToFrames(nn.Module):
    """
    it is for torch tensor
    """

    def __init__(self, n_samples, F=512, stride=256):
        super().__init__()

        assert((n_samples-F) % stride == 0)
        self.n_samples = n_samples
        self.n_frames = (n_samples - F) // stride + 1
        self.idx_mat = torch.empty((self.n_frames, F), dtype=torch.long)
        start = 0
        for i in range(self.n_frames):
            self.idx_mat[i, :] = torch.arange(start, start+F)
            start += stride


    def forward(self, sig):
        """
            sig: [B, 1, n_samples]
            return: [B, 1, nframes, F]
        """
        return sig[:, :, self.idx_mat]

    def overlapAdd(self, input):
        """
            reverse the segementation process
            input [B, 1, n_frames, F]
            return [B, 1, n_samples]
        """

        output = torch.zeros((input.shape[0], input.shape[1], self.n_samples), device=input.device)
        for i in range(self.n_frames):
            output[:, :, self.idx_mat[i, :]] += input[:, :, i, :]

        return output

class ImprovedTransformer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
    """

    def __init__(self, d_model, nhead=4, dropout=0, bidirectional=False):
        super(ImprovedTransformer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model

        # dff = 4 * d
        self.gru = nn.GRU(d_model, d_model * 2 * (int(not bidirectional) + 1), dropout=dropout, bidirectional=bidirectional)
        self.linear = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.functional.relu

    def forward(self, src):
        # type: (Tensor) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
        Shape:
            see the docs in Transformer class.
        """
        skip = src
        out, _ = self.self_attn(src, src, src, need_weights=False)
        out = self.norm1(skip + out)
        skip = out

        out, _ = self.gru(out)
        out = self.linear(self.activation(out))
        out = self.norm2(skip + out)

        return out


class DPTBlock(nn.Module):

    def __init__(self, n_channels, dropout=0, bidirectional=False):
        super(DPTBlock, self).__init__()

        # dual-path transformers

        self.within_trans = ImprovedTransformer(n_channels, nhead=4, dropout=dropout, bidirectional=bidirectional)
        self.within_norm = nn.GroupNorm(1, n_channels)

        self.across_trans = ImprovedTransformer(n_channels, nhead=4, dropout=dropout, bidirectional=bidirectional)
        self.across_norm = nn.GroupNorm(1, n_channels)

    def forward(self, input):
        #  input ---
        #  [b,  c,  num_segments, segment_window]

        b, c, num_segments, segment_window = input.shape

        within_output = input.permute(0, 2, 3, 1).contiguous().view(b*num_segments , segment_window, c)
        within_output = self.within_trans(within_output)  # [b*num_segments, segment_window, c]
        within_output = within_output.view(b, num_segments, segment_window, c).permute(0, 3, 1, 2 ).contiguous()  # [b, c, num_segments, segment_window]
        within_output = self.within_norm(within_output)
        within_output = input + within_output  # [b, c, num_segments, segment_window]


        across_output = within_output.permute(0, 3, 2, 1).contiguous().view(b*segment_window, num_segments, c)
        across_output = self.across_trans(across_output)  # [b*segment_window, num_segments, c]
        across_output = across_output.view(b, segment_window, num_segments, c).permute(0, 3, 2, 1).contiguous()  # [b, c, num_segments, segment_window]
        across_output = self.across_norm(across_output)
        across_output = across_output + within_output # [b, c, num_segments, segment_window]

        return across_output


class TransformerModule(nn.Module):
    def __init__(self, n_channels, n_block, dropout=0, bidirectional=False):
        super().__init__()
        self.n_block = n_block
        self.before = nn.Sequential(
            nn.Conv2d(n_channels, n_channels//2, kernel_size=1),
            nn.PReLU(n_channels//2)
        )

        self.blocks = nn.ModuleList([])
        for n in range(n_block):
            self.blocks.append(DPTBlock(n_channels//2, dropout, bidirectional))

        # conv2d and PReLU or
        # PReLU and conv2d ????
        self.after = nn.Sequential(
            nn.PReLU(n_channels//2),
            nn.Conv2d(n_channels//2, n_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.before(x)
        for n in range(self.n_block):
            x = self.blocks[n](x)

        return self.after(x)


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), padding='same')
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class DilatedDenseBlock(nn.Module):

    def __init__(self, input_size, n_layers=5, n_channels=64):
        super(DilatedDenseBlock, self).__init__()
        self.n_layers = n_layers
        kernel_size = (2, 3)
        self.block = nn.ModuleList([])

        # causal padding using copy or zero padding
        for n in range(n_layers):
            dil = 2**n
            # causal between frame
            bf_pad_length = dil * (kernel_size[0] - 1)
            # non causal within frames
            wf_pad_length = dil * (kernel_size[1] - 1)//2

            layer = nn.Sequential(
                                  nn.ConstantPad2d((wf_pad_length, wf_pad_length, 0, bf_pad_length), value=0.),
                                  nn.Conv2d(n_channels * (n + 1), n_channels, kernel_size=kernel_size, dilation=dil),
                                  nn.LayerNorm(input_size),
                                  nn.PReLU(n_channels)
                                  )
            self.block.append(layer)

    def forward(self, x):
        skip = x
        for n in range(self.n_layers):
            out = self.block[n](skip)
            skip = torch.cat([out, skip], dim=1)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, n_encode_channels, segment_window):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, n_encode_channels, kernel_size=(1, 1)),
            nn.LayerNorm(segment_window),
            nn.PReLU(n_encode_channels),

            DilatedDenseBlock(segment_window, 4, n_encode_channels),
            # [b, n_encode_channels, num_segments, segment_window//2]
            nn.Conv2d(n_encode_channels, n_encode_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.LayerNorm(segment_window//2),
            nn.PReLU(n_encode_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class MaskingModule(nn.Module):
    def __init__(self, n_channels):
        super().__init__()


        self.tanh_branch = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1),
            nn.Tanh()
        )
        self.sigmoid_branch = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.mask_out = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1),
            nn.ReLU(inplace=True))

    def forward(self, x):

        t = self.tanh_branch(x)
        s = self.sigmoid_branch(x)
        output = self.mask_out(t*s)
        return output


class Decoder(nn.Module):
    def __init__(self, out_channels, n_encode_channels, segment_window):
        super().__init__()
        self.decoder = nn.Sequential(
            DilatedDenseBlock(segment_window//2, 4, n_encode_channels),
            SPConvTranspose2d(n_encode_channels, n_encode_channels, kernel_size=(1, 3), r=2),
            nn.LayerNorm(segment_window),
            nn.PReLU(n_encode_channels),
            nn.Conv2d(n_encode_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        return self.decoder(x)


class TSTNN(nn.Module):
    def __init__(self,
                 num_samples,
                 segment_window=512,
                 segment_stride=256,
                 n_encode_channels=64):
        super(TSTNN, self).__init__()

        # in out channels are fixed to be 1: single channel audio
        in_channels = 1
        out_channels = 1

        self.segment = SignalToFrames(num_samples, segment_window, segment_stride)

        self.encoder = Encoder(in_channels, n_encode_channels, segment_window)

        self.dual_transformer = TransformerModule(n_encode_channels, 4)

        self.maskModule = MaskingModule(n_encode_channels)
        # gated output layer

        self.decoder = Decoder(out_channels, n_encode_channels, segment_window)

    def forward(self, x):
        """
            x: [B, 1, T]
        """

        # concat and segement
        x = self.segment(x) # [B, 1, num_segments, segment_window]
        x = self.encoder(x) # [B, c, num_segments, segment_window//2]

        mask = self.dual_transformer(x) # [B, c, num_segments, segment_window//2]
        mask = self.maskModule(mask) # [B, c, num_segments, segment_window//2]

        out = x * mask
        out = self.decoder(out)

        return self.segment.overlapAdd(out)


if __name__ == '__main__':
    input = torch.Tensor([[[1,2,3,4,5,6,7,8,9,10]]])
    segment = SignalToFrames(input.shape[-1], 4, 2)
    segmented = segment(input)
    print(segmented)
    sig = segment.overlapAdd(segmented)
    print(sig)



