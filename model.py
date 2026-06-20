import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, atom_in_fea.shape[1]),
            atom_nbr_fea, nbr_fea
        ], dim=2)
        
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, total_gated_fea.shape[2])).view(N, M, -1)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class SimpleCGCNN(nn.Module):
    def __init__(self, orig_atom_fea_len=92, nbr_fea_len=41, atom_fea_len=64, h_fea_len=128):
        super(SimpleCGCNN, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.conv = ConvLayer(atom_fea_len, nbr_fea_len)
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.fc_out = nn.Linear(h_fea_len, 1)
        self.softplus = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        atom_fea = self.embedding(atom_fea)
        atom_fea = self.conv(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.softplus(self.conv_to_fc(atom_fea.mean(dim=0)))
        return self.fc_out(crys_fea)