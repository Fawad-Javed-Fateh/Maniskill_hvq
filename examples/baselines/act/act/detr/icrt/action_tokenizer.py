from act.detr.icrt.lipfq_vae import LipFQ_VAE
from act.detr.icrt.hvq_vae import HierarchicalLFQHVQVAE
from act.detr.icrt.lfq_vae import LFQ_VAE
from act.detr.icrt.mlp import MLP_Encoder
from act.detr.icrt.bin import AdaptiveBinActionEmbedding

def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder = LipFQ_VAE(feature_dim=d_model, latent_dim=d_model)

    return encoder

def build_lfq_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder = HierarchicalLFQHVQVAE(feature_dim=d_model, z_dim=d_model,q_dim=d_model,num_z_codes=64,num_q_codes=16)

    return encoder

def build_hvq_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder = LFQ_VAE(feature_dim=d_model, latent_dim=d_model)

    return encoder

def build_mega_mind(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder = LFQ_VAE(feature_dim=d_model, latent_dim=d_model)

    return encoder


def build_mlp_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder = MLP_Encoder(feature_dim=d_model, latent_dim=d_model)

    return encoder


def build_bin_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder = AdaptiveBinActionEmbedding(action_dim=d_model, output_dim=d_model)

    return encoder