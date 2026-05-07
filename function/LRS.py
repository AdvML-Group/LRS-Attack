import torch
import torch.nn as nn
import torch.nn.functional as F


def inv3_logit(model, x):
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)
    x = F.avg_pool2d(x, kernel_size=8)
    x = F.dropout(x, training=False)
    x = x.view(x.size(0), -1)
    x = model.last_linear(x)
    return x


def multi_lrs_inv3(model, input, num_iters,
                    compression_rate_shallow, rank_ratio_shallow,
                    compression_rate_balanced, rank_ratio_balanced,
                    compression_rate_deep, rank_ratio_deep):
    # shallow
    x_sparse = model.Conv2d_1a_3x3(input)
    x_sparse = model.Conv2d_2a_3x3(x_sparse)
    x_sparse = model.Conv2d_2b_3x3(x_sparse)
    x_sparse = F.max_pool2d(x_sparse, kernel_size=3, stride=2)
    x_sparse = model.Conv2d_3b_1x1(x_sparse)
    x_sparse = model.Conv2d_4a_3x3(x_sparse)
    x_sparse = F.max_pool2d(x_sparse, kernel_size=3, stride=2)
    x_sparse = model.Mixed_5b(x_sparse)

    B, C, H, W = x_sparse.size()
    d_out = C
    d_in = H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_shallow, rank_ratio_shallow)
    feat_sparse = x_sparse.view(B, C, H * W).float()
    D_sparse = torch.sqrt(torch.sum(feat_sparse * feat_sparse, dim=-1, keepdim=True))
    normalized_feat_sparse = feat_sparse / (D_sparse + 1e-8)

    # sparse
    _, sparse_comp = altern_ls(
        normalized_feat_sparse,
        num_iters,
        target_rank,
        num_nonzeros=num_nonzeros
    )
    sparse_comp = sparse_comp * D_sparse
    x_sparse_new = sparse_comp.view(B, C, H, W)

    x_sparse_new = model.Mixed_5c(x_sparse_new)
    x_sparse_new = model.Mixed_5d(x_sparse_new)
    x_sparse_new = model.Mixed_6a(x_sparse_new)
    x_sparse_new = model.Mixed_6b(x_sparse_new)
    x_sparse_new = model.Mixed_6c(x_sparse_new)
    x_sparse_new = model.Mixed_6d(x_sparse_new)
    x_sparse_new = model.Mixed_6e(x_sparse_new)
    logit_sparse = inv3_logit(model, x_sparse_new)

    # balanced
    x_balanced = model.Conv2d_1a_3x3(input)
    x_balanced = model.Conv2d_2a_3x3(x_balanced)
    x_balanced = model.Conv2d_2b_3x3(x_balanced)
    x_balanced = F.max_pool2d(x_balanced, kernel_size=3, stride=2)
    x_balanced = model.Conv2d_3b_1x1(x_balanced)
    x_balanced = model.Conv2d_4a_3x3(x_balanced)
    x_balanced = F.max_pool2d(x_balanced, kernel_size=3, stride=2)
    x_balanced = model.Mixed_5b(x_balanced)
    x_balanced = model.Mixed_5c(x_balanced)
    x_balanced = model.Mixed_5d(x_balanced)

    B, C, H, W = x_balanced.size()
    d_out = C
    d_in = H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_balanced, rank_ratio_balanced)
    feat_balanced = x_balanced.view(B, C, H * W).float()
    D_balanced = torch.sqrt(torch.sum(feat_balanced * feat_balanced, dim=-1, keepdim=True))
    normalized_feat_balanced = feat_balanced / (D_balanced + 1e-8)

    low_rank_comp_balanced, sparse_comp_balanced = altern_ls(
        normalized_feat_balanced,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros
    )


    decomp_balanced = (low_rank_comp_balanced + sparse_comp_balanced) * D_balanced / 2
    x_balanced_new = decomp_balanced.view(B, C, H, W)

    x_balanced_new = model.Mixed_6a(x_balanced_new)
    x_balanced_new = model.Mixed_6b(x_balanced_new)
    x_balanced_new = model.Mixed_6c(x_balanced_new)
    x_balanced_new = model.Mixed_6d(x_balanced_new)
    x_balanced_new = model.Mixed_6e(x_balanced_new)
    logit_balanced = inv3_logit(model, x_balanced_new)

    # deep
    x_lowrank = model.Conv2d_1a_3x3(input)
    x_lowrank = model.Conv2d_2a_3x3(x_lowrank)
    x_lowrank = model.Conv2d_2b_3x3(x_lowrank)
    x_lowrank = F.max_pool2d(x_lowrank, kernel_size=3, stride=2)
    x_lowrank = model.Conv2d_3b_1x1(x_lowrank)
    x_lowrank = model.Conv2d_4a_3x3(x_lowrank)
    x_lowrank = F.max_pool2d(x_lowrank, kernel_size=3, stride=2)
    x_lowrank = model.Mixed_5b(x_lowrank)
    x_lowrank = model.Mixed_5c(x_lowrank)
    x_lowrank = model.Mixed_5d(x_lowrank)
    x_lowrank = model.Mixed_6a(x_lowrank)
    x_lowrank = model.Mixed_6b(x_lowrank)
    x_lowrank = model.Mixed_6c(x_lowrank)
    x_lowrank = model.Mixed_6d(x_lowrank)
    x_lowrank = model.Mixed_6e(x_lowrank)

    B, C, H, W = x_lowrank.size()
    d_out = C
    d_in = H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_deep, rank_ratio_deep)
    feat_lowrank = x_lowrank.view(B, C, H * W).float()
    D_lowrank = torch.sqrt(torch.sum(feat_lowrank * feat_lowrank, dim=-1, keepdim=True))
    normalized_feat_lowrank = feat_lowrank / (D_lowrank + 1e-8)

    low_rank_comp, _ = altern_ls(
        normalized_feat_lowrank,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros
    )

    low_rank_comp = low_rank_comp * D_lowrank
    x_lowrank_new = low_rank_comp.view(B, C, H, W)
    logit_lowrank = inv3_logit(model, x_lowrank_new)

    # original
    x_ori = model.Conv2d_1a_3x3(input)
    x_ori = model.Conv2d_2a_3x3(x_ori)
    x_ori = model.Conv2d_2b_3x3(x_ori)
    x_ori = F.max_pool2d(x_ori, kernel_size=3, stride=2)
    x_ori = model.Conv2d_3b_1x1(x_ori)
    x_ori = model.Conv2d_4a_3x3(x_ori)
    x_ori = F.max_pool2d(x_ori, kernel_size=3, stride=2)
    x_ori = model.Mixed_5b(x_ori)
    x_ori = model.Mixed_5c(x_ori)
    x_ori = model.Mixed_5d(x_ori)
    x_ori = model.Mixed_6a(x_ori)
    x_ori = model.Mixed_6b(x_ori)
    x_ori = model.Mixed_6c(x_ori)
    x_ori = model.Mixed_6d(x_ori)
    x_ori = model.Mixed_6e(x_ori)
    logit_ori = inv3_logit(model, x_ori)

    return (logit_sparse + logit_balanced + logit_lowrank + logit_ori) / 4


def calculate_lrs_parameters(d_out, d_in, compression_rate, rank_ratio):
    r = int(rank_ratio * (1 - compression_rate) * (d_out * d_in) / (d_out + d_in))
    k = int((1 - rank_ratio) * (1 - compression_rate) * d_out * d_in)
    return r, k

def approx_low_rank(A, rank, n_iter=3, reg_lambda=1e-4, use_warm_start=True, prev_U=None, prev_V=None):
    B, C, HW = A.shape
    device = A.device

    # warm start
    if use_warm_start and prev_U is not None and prev_V is not None:

        if prev_U.shape == (B, C, rank) and prev_V.shape == (B, HW, rank):

            U = prev_U + torch.randn_like(prev_U) * 0.01
            V = prev_V + torch.randn_like(prev_V) * 0.01

            U = torch.linalg.qr(U).Q
            V = torch.linalg.qr(V).Q
        else:

            V = torch.randn(B, HW, rank, device=device)
            V = torch.linalg.qr(V).Q
            U = None
    else:

        V = torch.randn(B, HW, rank, device=device)
        V = torch.linalg.qr(V).Q
        U = None

    prev_loss = None

    for i in range(n_iter):

        if U is None:
            U = torch.matmul(A, V)
        else:
            U = torch.matmul(A, V) - reg_lambda * U

        U = torch.linalg.qr(U).Q

        V = torch.matmul(A.transpose(1, 2), U) - reg_lambda * V
        V = torch.linalg.qr(V).Q

        approx = torch.bmm(U, V.transpose(1, 2))
        loss = torch.norm(A - approx, p='fro').item()
        if prev_loss is not None and abs(loss - prev_loss) < 1e-5:
            break
        prev_loss = loss

    S = torch.bmm(U.transpose(1, 2), torch.bmm(A, V))
    S = S.diagonal(dim1=-2, dim2=-1)

    return U, S, V


def altern_ls(weight, num_iters, target_rank, num_nonzeros):
    device = weight.device
    B, C, HW = weight.shape
    sparse_component = torch.zeros_like(weight, device=device)

    prev_U = None
    prev_V = None

    keep_topk = HW - num_nonzeros // C
    keep_topk = max(1, min(keep_topk, HW))
    for _ in range(num_iters):

        residual = weight - sparse_component
        U, S, V = approx_low_rank(
            residual,
            rank=target_rank,
            n_iter=3,
            reg_lambda=1e-4,
            use_warm_start=True,
            prev_U=prev_U,
            prev_V=prev_V
        )

        prev_U = U.clone()
        prev_V = V.clone()

        S_truncated = S.clone()
        if S_truncated.shape[-1] > target_rank:
            S_truncated[:, target_rank:] = 0

        low_rank_component = torch.matmul(
            torch.matmul(U, torch.diag_embed(S_truncated)),
            V.transpose(1, 2)
        )

        sparse_component = weight - low_rank_component

        abs_values = torch.abs(sparse_component)
        values, _ = torch.topk(abs_values, k=keep_topk, dim=2, largest=True)
        thresholds = values[:, :, -1:]
        mask = abs_values < thresholds
        sparse_component[mask] = 0

    return low_rank_component, sparse_component


def inv4_logit(model, x):
    x = model.features[18](x)
    x = model.features[19](x)
    x = model.features[20](x)
    x = model.features[21](x)
    x = F.avg_pool2d(x, kernel_size=8)
    x = F.dropout(x, training=False)
    x = x.view(x.size(0), -1)
    x = model.last_linear(x)
    return x


def multi_lrs_inv4(model, input, num_iters=5,
                    compression_rate_shallow=0.8, rank_ratio_shallow=0.01,
                    compression_rate_balanced=0.5, rank_ratio_balanced=0.04,
                    compression_rate_deep=0, rank_ratio_deep=0.1):

    x_shallow = model.features[0](input)
    x_shallow = model.features[1](x_shallow)
    x_shallow = model.features[2](x_shallow)
    x_shallow = model.features[3](x_shallow)
    x_shallow = model.features[4](x_shallow)
    x_shallow = model.features[5](x_shallow)

    B, C, H, W = x_shallow.size()
    d_out = C
    d_in = H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(
        d_out, d_in, compression_rate=compression_rate_shallow, rank_ratio=rank_ratio_shallow
    )
    target_rank = max(1, target_rank)
    num_nonzeros = max(0, num_nonzeros)

    feat_shallow = x_shallow.view(B, C, H * W).float()
    D_shallow = torch.sqrt(torch.sum(feat_shallow * feat_shallow, dim=-1, keepdim=True))
    D_shallow = torch.clamp(D_shallow, min=1e-8)
    normalized_feat_shallow = feat_shallow / D_shallow

    _, sparse_comp_shallow = altern_ls(
        normalized_feat_shallow,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros,
    )
    sparse_comp_shallow = sparse_comp_shallow * D_shallow
    x_shallow_new = sparse_comp_shallow.view(B, C, H, W)

    for i in range(6, 18):
        x_shallow_new = model.features[i](x_shallow_new)
    logit_shallow = inv4_logit(model, x_shallow_new)

    x_balanced = model.features[0](input)
    x_balanced = model.features[1](x_balanced)
    x_balanced = model.features[2](x_balanced)
    x_balanced = model.features[3](x_balanced)
    x_balanced = model.features[4](x_balanced)
    x_balanced = model.features[5](x_balanced)
    for i in range(6, 10):
        x_balanced = model.features[i](x_balanced)

    B, C, H, W = x_balanced.size()
    d_out = C
    d_in = H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(
        d_out, d_in, compression_rate=compression_rate_balanced, rank_ratio=rank_ratio_balanced
    )
    target_rank = max(1, target_rank)
    num_nonzeros = max(0, num_nonzeros)

    feat_balanced = x_balanced.view(B, C, H * W).float()
    D_balanced = torch.sqrt(torch.sum(feat_balanced * feat_balanced, dim=-1, keepdim=True))
    D_balanced = torch.clamp(D_balanced, min=1e-8)
    normalized_feat_balanced = feat_balanced / D_balanced

    low_rank_comp_balanced, sparse_comp_balanced = altern_ls(
        normalized_feat_balanced,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros,
    )
    decomp_balanced = (low_rank_comp_balanced + sparse_comp_balanced) * D_balanced / 2
    x_balanced_new = decomp_balanced.view(B, C, H, W)

    for i in range(10, 18):
        x_balanced_new = model.features[i](x_balanced_new)
    logit_balanced = inv4_logit(model, x_balanced_new)

    x_lowrank = model.features[0](input)
    x_lowrank = model.features[1](x_lowrank)
    x_lowrank = model.features[2](x_lowrank)
    x_lowrank = model.features[3](x_lowrank)
    x_lowrank = model.features[4](x_lowrank)
    x_lowrank = model.features[5](x_lowrank)
    for i in range(6, 18):
        x_lowrank = model.features[i](x_lowrank)

    B, C, H, W = x_lowrank.size()
    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_deep, rank_ratio_deep)

    feat_lowrank = x_lowrank.view(B, C, H * W).float()
    D_lowrank = torch.sqrt(torch.sum(feat_lowrank * feat_lowrank, dim=-1, keepdim=True))
    D_lowrank = torch.clamp(D_lowrank, min=1e-8)
    normalized_feat_lowrank = feat_lowrank / D_lowrank

    low_rank_comp, _ = altern_ls(
        normalized_feat_lowrank,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros,
    )
    low_rank_comp = low_rank_comp * D_lowrank
    x_lowrank_new = low_rank_comp.view(B, C, H, W)
    logit_lowrank = inv4_logit(model, x_lowrank_new)

    x_ori = model.features[0](input)
    x_ori = model.features[1](x_ori)
    x_ori = model.features[2](x_ori)
    x_ori = model.features[3](x_ori)
    x_ori = model.features[4](x_ori)
    x_ori = model.features[5](x_ori)
    for i in range(6, 18):
        x_ori = model.features[i](x_ori)
    logit_ori = inv4_logit(model, x_ori)

    combined_logit = (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4
    return combined_logit


def incresv2_logit(model, x):
    x = model.mixed_7a(x)
    x = model.repeat_2(x)
    x = model.block8(x)
    x = model.conv2d_7b(x)
    x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
    x = F.dropout(x, training=False)
    x = x.view(x.size(0), -1)
    x = model.last_linear(x)
    return x


def multi_lrs_incresv2(model, input, num_iters=5,
                        compression_rate_shallow=0.8, rank_ratio_shallow=0.01,
                        compression_rate_balanced=0.5, rank_ratio_balanced=0.04,
                        compression_rate_deep=0, rank_ratio_deep=0.1):

    x_shallow = model.conv2d_1a(input)
    x_shallow = model.conv2d_2a(x_shallow)
    x_shallow = model.conv2d_2b(x_shallow)
    x_shallow = model.maxpool_3a(x_shallow)
    x_shallow = model.conv2d_3b(x_shallow)
    x_shallow = model.conv2d_4a(x_shallow)
    x_shallow = model.maxpool_5a(x_shallow)
    x_shallow = model.mixed_5b(x_shallow)

    B, C, H, W = x_shallow.size()
    d_out, d_in = C, H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_shallow, rank_ratio_shallow)

    feat = x_shallow.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)

    _, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    x_shallow_new = (sparse_comp * D).view(B, C, H, W)

    x_shallow_new = model.repeat(x_shallow_new)
    x_shallow_new = model.mixed_6a(x_shallow_new)
    x_shallow_new = model.repeat_1(x_shallow_new)
    logit_shallow = incresv2_logit(model, x_shallow_new)

    x_balanced = model.conv2d_1a(input)
    x_balanced = model.conv2d_2a(x_balanced)
    x_balanced = model.conv2d_2b(x_balanced)
    x_balanced = model.maxpool_3a(x_balanced)
    x_balanced = model.conv2d_3b(x_balanced)
    x_balanced = model.conv2d_4a(x_balanced)
    x_balanced = model.maxpool_5a(x_balanced)
    x_balanced = model.mixed_5b(x_balanced)
    x_balanced = model.repeat(x_balanced)

    B, C, H, W = x_balanced.size()
    d_out, d_in = C, H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_balanced, rank_ratio_balanced)

    feat = x_balanced.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)

    low_rank_comp, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    decomp = (low_rank_comp + sparse_comp) * D / 2
    x_balanced_new = decomp.view(B, C, H, W)

    x_balanced_new = model.mixed_6a(x_balanced_new)
    x_balanced_new = model.repeat_1(x_balanced_new)
    logit_balanced = incresv2_logit(model, x_balanced_new)

    x_lowrank = model.conv2d_1a(input)
    x_lowrank = model.conv2d_2a(x_lowrank)
    x_lowrank = model.conv2d_2b(x_lowrank)
    x_lowrank = model.maxpool_3a(x_lowrank)
    x_lowrank = model.conv2d_3b(x_lowrank)
    x_lowrank = model.conv2d_4a(x_lowrank)
    x_lowrank = model.maxpool_5a(x_lowrank)
    x_lowrank = model.mixed_5b(x_lowrank)
    x_lowrank = model.repeat(x_lowrank)
    x_lowrank = model.mixed_6a(x_lowrank)
    x_lowrank = model.repeat_1(x_lowrank)

    B, C, H, W = x_lowrank.size()
    target_rank, num_nonzeros = calculate_lrs_parameters(C, H * W, compression_rate_deep, rank_ratio_deep)

    feat = x_lowrank.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)

    low_rank_comp, _ = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    x_lowrank_new = (low_rank_comp * D).view(B, C, H, W)
    logit_lowrank = incresv2_logit(model, x_lowrank_new)

    x_ori = model.conv2d_1a(input)
    x_ori = model.conv2d_2a(x_ori)
    x_ori = model.conv2d_2b(x_ori)
    x_ori = model.maxpool_3a(x_ori)
    x_ori = model.conv2d_3b(x_ori)
    x_ori = model.conv2d_4a(x_ori)
    x_ori = model.maxpool_5a(x_ori)
    x_ori = model.mixed_5b(x_ori)
    x_ori = model.repeat(x_ori)
    x_ori = model.mixed_6a(x_ori)
    x_ori = model.repeat_1(x_ori)
    logit_ori = incresv2_logit(model, x_ori)

    return (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4


def vgg16_logit(model, x, start_idx):

    if hasattr(model, '_features'):
        features = list(model._features.children())
    else:
        features = list(model.features.children())

    for layer in features[start_idx:]:
        x = layer(x)
    x = F.adaptive_avg_pool2d(x, (7, 7))
    x = torch.flatten(x, 1)

    if hasattr(model, 'classifier'):
        x = model.classifier(x)
    else:
        x = model.linear0(x)
        x = model.relu0(x)
        x = model.dropout0(x)
        x = model.linear1(x)
        x = model.relu1(x)
        x = model.dropout1(x)
        x = model.last_linear(x)
    return x


def multi_lrs_vgg16(model, input, num_iters=5, shallow_idx=10, balanced_idx=16, deep_idx=23,
                     compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                     compression_rate_balanced=0.2, rank_ratio_balanced=0.02,
                     compression_rate_deep=0, rank_ratio_deep=0.1):

    def forward_to(model, x, end_idx):

        if hasattr(model, '_features'):
            features = list(model._features.children())
        else:
            features = list(model.features.children())
        for layer in features[:end_idx]:
            x = layer(x)
        return x

    x_shallow = forward_to(model, input, shallow_idx)
    B, C, H, W = x_shallow.size()
    d_out, d_in = C, H * W
    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_shallow, rank_ratio_shallow)
    feat = x_shallow.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)
    _, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    x_shallow_new = (sparse_comp * D).view(B, C, H, W)

    features = list(model._features.children()) if hasattr(model, '_features') else list(model.features.children())
    for layer in features[shallow_idx:deep_idx]:
        x_shallow_new = layer(x_shallow_new)
    logit_shallow = vgg16_logit(model, x_shallow_new, start_idx=deep_idx)

    x_balanced = forward_to(model, input, balanced_idx)
    B, C, H, W = x_balanced.size()
    d_out, d_in = C, H * W
    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_balanced, rank_ratio_balanced)
    feat = x_balanced.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)
    low_rank_comp, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    decomp = (low_rank_comp + sparse_comp) * D / 2
    x_balanced_new = decomp.view(B, C, H, W)
    features = list(model._features.children()) if hasattr(model, '_features') else list(model.features.children())
    for layer in features[balanced_idx:deep_idx]:
        x_balanced_new = layer(x_balanced_new)
    logit_balanced = vgg16_logit(model, x_balanced_new, start_idx=deep_idx)

    x_deep = forward_to(model, input, deep_idx)
    B, C, H, W = x_deep.size()
    target_rank, num_nonzeros = calculate_lrs_parameters(C, H * W, compression_rate_deep, rank_ratio_deep)
    feat = x_deep.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)
    low_rank_comp, _ = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    x_deep_new = (low_rank_comp * D).view(B, C, H, W)
    logit_lowrank = vgg16_logit(model, x_deep_new, start_idx=deep_idx)

    x_ori = forward_to(model, input, deep_idx)
    logit_ori = vgg16_logit(model, x_ori, start_idx=deep_idx)

    return (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4


def vgg19_logit(model, x, start_idx):

    if hasattr(model, '_features'):
        features = list(model._features.children())
    else:
        features = list(model.features.children())
    
    for layer in features[start_idx:]:
        x = layer(x)

    if hasattr(model, 'avgpool'):
        x = model.avgpool(x)
    else:
        x = F.adaptive_avg_pool2d(x, (7, 7))
    x = torch.flatten(x, 1)

    if hasattr(model, 'classifier'):
        x = model.classifier(x)
    else:
        x = model.linear0(x)
        x = model.relu0(x)
        x = model.dropout0(x)
        x = model.linear1(x)
        x = model.relu1(x)
        x = model.dropout1(x)
        x = model.last_linear(x)
    return x

def multi_lrs_vgg19(model, input, num_iters=5, shallow_idx=8, balanced_idx=17, deep_idx=25,
                     compression_rate_shallow=0.8, rank_ratio_shallow=0.01,
                     compression_rate_balanced=0.5, rank_ratio_balanced=0.04,
                     compression_rate_deep=0, rank_ratio_deep=0.1):


    def forward_to(model, x, end_idx):

        if hasattr(model, '_features'):
            features = list(model._features.children())
        else:
            features = list(model.features.children())
        for layer in features[:end_idx]:
            x = layer(x)
        return x

    # Shallow
    x_shallow = forward_to(model, input, shallow_idx)
    B, C, H, W = x_shallow.size()
    d_out, d_in = C, H * W
    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_shallow, rank_ratio_shallow)
    feat = x_shallow.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)
    _, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    x_shallow_new = (sparse_comp * D).view(B, C, H, W)
    features = list(model._features.children()) if hasattr(model, '_features') else list(model.features.children())
    for layer in features[shallow_idx:deep_idx]:
        x_shallow_new = layer(x_shallow_new)
    logit_shallow = vgg19_logit(model, x_shallow_new, start_idx=deep_idx)

    # Balanced
    x_balanced = forward_to(model, input, balanced_idx)
    B, C, H, W = x_balanced.size()
    d_out, d_in = C, H * W
    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_balanced, rank_ratio_balanced)
    feat = x_balanced.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)
    low_rank_comp, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    decomp = (low_rank_comp + sparse_comp) * D / 2
    x_balanced_new = decomp.view(B, C, H, W)
    features = list(model._features.children()) if hasattr(model, '_features') else list(model.features.children())
    for layer in features[balanced_idx:deep_idx]:
        x_balanced_new = layer(x_balanced_new)
    logit_balanced = vgg19_logit(model, x_balanced_new, start_idx=deep_idx)

    # Deep
    x_deep = forward_to(model, input, deep_idx)
    B, C, H, W = x_deep.size()
    target_rank, num_nonzeros = calculate_lrs_parameters(C, H * W, compression_rate_deep, rank_ratio_deep)
    feat = x_deep.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    normalized = feat / (D + 1e-8)
    low_rank_comp, _ = altern_ls(normalized, num_iters, target_rank, num_nonzeros)
    x_deep_new = (low_rank_comp * D).view(B, C, H, W)
    logit_lowrank = vgg19_logit(model, x_deep_new, start_idx=deep_idx)

    # Original
    x_ori = forward_to(model, input, deep_idx)
    logit_ori = vgg19_logit(model, x_ori, start_idx=deep_idx)

    return (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4



def vit_logit(model, x, target_layer):
    for i in range(target_layer, len(model.blocks)):
        x = model.blocks[i](x)
    x = model.norm(x)
    if getattr(model, 'cls_token', None) is not None:
        x = x[:, 0]
    return model.head(x)


def decompose_qkv_separate(qkv_output, num_iters, compression_rate, rank_ratio):

    B, N, C3 = qkv_output.shape
    C = C3 // 3

    qkv_reshaped = qkv_output.view(B, N, 3, C)  # [B, N, 3, C]
    q = qkv_reshaped[:, :, 0, :]  # [B, N, C]
    k = qkv_reshaped[:, :, 1, :]  # [B, N, C]
    v = qkv_reshaped[:, :, 2, :]  # [B, N, C]

    decomposed_list = []
    for mat in [q, k, v]:

        feat = mat.transpose(1, 2)

        D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
        D = torch.clamp(D, min=1e-8)
        normalized = feat / D

        target_rank, num_nonzeros = calculate_lrs_parameters(C, N, compression_rate, rank_ratio)


        low_rank_comp, sparse_comp = altern_ls(normalized, num_iters, target_rank, num_nonzeros)

        decomposed = (low_rank_comp + sparse_comp) * D / 2
        decomposed = decomposed.transpose(1, 2)  # [B, N, C]

        decomposed_list.append(decomposed)

    qkv_decomposed = torch.stack(decomposed_list, dim=2)  # [B, N, 3, C]
    qkv_decomposed = qkv_decomposed.view(B, N, 3 * C)  # [B, N, 3*C]

    return qkv_decomposed


def multi_lrs_vit_qkv(model, input, num_iters=5, shallow_layer=4, balanced_layer=7, deep_layer=10,
                       compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                       compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                       compression_rate_deep=0, rank_ratio_deep=0.1):

    x0 = model.patch_embed(input)
    if getattr(model, 'cls_token', None) is not None:
        cls_tokens = model.cls_token.expand(x0.shape[0], -1, -1)
        x0 = torch.cat((cls_tokens, x0), dim=1)
    x0 = model.pos_drop(x0 + model.pos_embed)

    def run_blocks(x, k):
        for i in range(k):
            x = model.blocks[i](x)
        return x

    def apply_qkv_decomposition(x, block, compression_rate, rank_ratio):

        x_norm = block.norm1(x)

        qkv = block.attn.qkv(x_norm)  # [B, N, 3*C]

        qkv_decomposed = decompose_qkv_separate(qkv, num_iters, compression_rate, rank_ratio)

        B, N, _ = qkv_decomposed.shape
        C = model.embed_dim
        qkv_reshaped = qkv_decomposed.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
        qkv_reshaped = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]

        attn = (q @ k.transpose(-2, -1)) * block.attn.scale
        attn = attn.softmax(dim=-1)
        attn = block.attn.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = block.attn.proj(x_attn)
        x_attn = block.attn.proj_drop(x_attn)

        x = x + x_attn

        if hasattr(block, 'drop_path'):
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        else:
            x = x + block.mlp(block.norm2(x))

        return x

    # Shallow
    x_shallow = run_blocks(x0.clone(), shallow_layer - 1) if shallow_layer > 0 else x0.clone()
    x_shallow = apply_qkv_decomposition(
        x_shallow, model.blocks[shallow_layer],
        compression_rate_shallow, rank_ratio_shallow
    )
    for i in range(shallow_layer + 1, len(model.blocks)):
        x_shallow = model.blocks[i](x_shallow)
    x_shallow = model.norm(x_shallow)
    logit_shallow = model.head(x_shallow[:, 0] if model.cls_token is not None else x_shallow.mean(1))
    del x_shallow

    # Balanced
    x_balanced = run_blocks(x0.clone(), balanced_layer - 1) if balanced_layer > 0 else x0.clone()
    x_balanced = apply_qkv_decomposition(
        x_balanced, model.blocks[balanced_layer],
        compression_rate_balanced, rank_ratio_balanced
    )
    for i in range(balanced_layer + 1, len(model.blocks)):
        x_balanced = model.blocks[i](x_balanced)
    x_balanced = model.norm(x_balanced)
    logit_balanced = model.head(x_balanced[:, 0] if model.cls_token is not None else x_balanced.mean(1))
    del x_balanced

    # Deep
    x_deep = run_blocks(x0.clone(), deep_layer - 1) if deep_layer > 0 else x0.clone()
    x_deep = apply_qkv_decomposition(
        x_deep, model.blocks[deep_layer],
        compression_rate_deep, rank_ratio_deep
    )
    for i in range(deep_layer + 1, len(model.blocks)):
        x_deep = model.blocks[i](x_deep)
    x_deep = model.norm(x_deep)
    logit_deep = model.head(x_deep[:, 0] if model.cls_token is not None else x_deep.mean(1))
    del x_deep

    # Original
    x_ori = run_blocks(x0.clone(), len(model.blocks))
    x_ori = model.norm(x_ori)
    logit_ori = model.head(x_ori[:, 0] if model.cls_token is not None else x_ori.mean(1))
    del x_ori

    result = (logit_shallow + logit_balanced + logit_deep + logit_ori) / 4

    del logit_shallow, logit_balanced, logit_deep, logit_ori
    torch.cuda.empty_cache()

    return result


def multi_lrs_vit(model, input, num_iters=5, shallow_layer=4, balanced_layer=7, deep_layer=10,
                   compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                   compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                   compression_rate_deep=0, rank_ratio_deep=0.1):

    x0 = model.patch_embed(input)
    if getattr(model, 'cls_token', None) is not None:
        cls_tokens = model.cls_token.expand(x0.shape[0], -1, -1)
        x0 = torch.cat((cls_tokens, x0), dim=1)
    x0 = model.pos_drop(x0 + model.pos_embed)

    def run_blocks(x, k):
        for i in range(k):
            x = model.blocks[i](x)
        return x

    x_shallow = run_blocks(x0.clone(), shallow_layer)
    B, N, C = x_shallow.shape
    d_out = C
    d_in = N

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_shallow, rank_ratio_shallow)

    feat_shallow = x_shallow.transpose(1, 2).float()

    D_shallow = torch.sqrt(torch.sum(feat_shallow * feat_shallow, dim=-1, keepdim=True))
    D_shallow = torch.clamp(D_shallow, min=1e-8)
    normalized_feat_shallow = feat_shallow / D_shallow

    _, sparse_comp_shallow = altern_ls(
        normalized_feat_shallow,
        num_iters,
        target_rank,
        num_nonzeros=num_nonzeros
    )

    sparse_comp_shallow = sparse_comp_shallow * D_shallow
    x_shallow_new = sparse_comp_shallow.transpose(1, 2)

    for i in range(shallow_layer, len(model.blocks)):
        x_shallow_new = model.blocks[i](x_shallow_new)

    x_shallow_final = model.norm(x_shallow_new)
    logit_shallow = model.head(
        x_shallow_final[:, 0] if getattr(model, 'cls_token', None) is not None else x_shallow_final.mean(1))
    del x_shallow, x_shallow_new, x_shallow_final

    x_balanced = run_blocks(x0.clone(), balanced_layer)
    B, N, C = x_balanced.shape
    d_out = C
    d_in = N

    target_rank, num_nonzeros = calculate_lrs_parameters(d_out, d_in, compression_rate_balanced, rank_ratio_balanced)

    feat_balanced = x_balanced.transpose(1, 2).float()
    D_balanced = torch.sqrt(torch.sum(feat_balanced * feat_balanced, dim=-1, keepdim=True))
    D_balanced = torch.clamp(D_balanced, min=1e-8)
    normalized_feat_balanced = feat_balanced / D_balanced

    low_rank_comp_balanced, sparse_comp_balanced = altern_ls(
        normalized_feat_balanced,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros
    )

    decomp_balanced = (low_rank_comp_balanced + sparse_comp_balanced) * D_balanced / 2
    x_balanced_new = decomp_balanced.transpose(1, 2)

    for i in range(balanced_layer, len(model.blocks)):
        x_balanced_new = model.blocks[i](x_balanced_new)

    x_balanced_final = model.norm(x_balanced_new)
    logit_balanced = model.head(
        x_balanced_final[:, 0] if getattr(model, 'cls_token', None) is not None else x_balanced_final.mean(1))
    del x_balanced, x_balanced_new, x_balanced_final

    x_deep = run_blocks(x0.clone(), deep_layer)
    B, N, C = x_deep.shape

    target_rank, num_nonzeros = calculate_lrs_parameters(C, N, compression_rate_deep, rank_ratio_deep)

    feat_deep = x_deep.transpose(1, 2).float()
    D_deep = torch.sqrt(torch.sum(feat_deep * feat_deep, dim=-1, keepdim=True))
    D_deep = torch.clamp(D_deep, min=1e-8)
    normalized_feat_deep = feat_deep / D_deep

    low_rank_comp, _ = altern_ls(
        normalized_feat_deep,
        num_iters,
        target_rank=target_rank,
        num_nonzeros=num_nonzeros
    )

    low_rank_comp = low_rank_comp * D_deep
    x_deep_new = low_rank_comp.transpose(1, 2)

    for i in range(deep_layer, len(model.blocks)):
        x_deep_new = model.blocks[i](x_deep_new)

    x_deep_final = model.norm(x_deep_new)
    logit_lowrank = model.head(
        x_deep_final[:, 0] if getattr(model, 'cls_token', None) is not None else x_deep_final.mean(1))
    del x_deep, x_deep_new, x_deep_final

    x_ori = run_blocks(x0.clone(), len(model.blocks))
    x_ori = model.norm(x_ori)
    logit_ori = model.head(x_ori[:, 0] if getattr(model, 'cls_token', None) is not None else x_ori.mean(1))
    del x_ori

    result = (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4

    del logit_shallow, logit_balanced, logit_lowrank, logit_ori
    torch.cuda.empty_cache()

    return result


def densenet_logit(model, x):

    x = model.features.norm5(x)
    x = F.relu(x, inplace=False)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = model.last_linear(x)
    return x


def multi_lrs_densenet(model, input, num_iters=5,
                        compression_rate_shallow=0.8, rank_ratio_shallow=0.01,
                        compression_rate_balanced=0.5, rank_ratio_balanced=0.04,
                        compression_rate_deep=0, rank_ratio_deep=0.1):

    def _disable_inplace_relu(m: nn.Module):
        for sub in m.modules():
            if isinstance(sub, nn.ReLU) and getattr(sub, 'inplace', False):
                sub.inplace = False

    _disable_inplace_relu(model)
    def common_forward(x):
        x = model.features.conv0(x)
        x = model.features.norm0(x)
        if hasattr(model.features, 'relu0'):
            x = model.features.relu0(x)
        elif hasattr(model.features, 'act0'):
            x = model.features.act0(x)
        else:
            x = F.relu(x, inplace=False)
        x = model.features.pool0(x)
        return x

    x_shallow = common_forward(input)
    x_shallow = model.features.denseblock1(x_shallow)  # Dense Block 1
    x_shallow = model.features.transition1(x_shallow)  # Transition 1

    B, C, H, W = x_shallow.size()
    d_out, d_in = C, H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(
        d_out, d_in, compression_rate_shallow, rank_ratio_shallow
    )

    feat = x_shallow.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    D = torch.clamp(D, min=1e-8)
    normalized = feat / D

    _, sparse_comp = altern_ls(
        normalized, num_iters, target_rank, num_nonzeros
    )
    x_shallow_new = (sparse_comp * D).view(B, C, H, W)

    x_shallow_new = model.features.denseblock2(x_shallow_new)  # Dense Block 2
    x_shallow_new = model.features.transition2(x_shallow_new)  # Transition 2
    x_shallow_new = model.features.denseblock3(x_shallow_new)  # Dense Block 3
    x_shallow_new = model.features.transition3(x_shallow_new)  # Transition 3

    x_shallow_final = model.features.denseblock4(x_shallow_new)  # Dense Block 4
    logit_shallow = densenet_logit(model, x_shallow_final)

    x_balanced = common_forward(input)
    x_balanced = model.features.denseblock1(x_balanced)  # Dense Block 1
    x_balanced = model.features.transition1(x_balanced)  # Transition 1
    x_balanced = model.features.denseblock2(x_balanced)  # Dense Block 2
    x_balanced = model.features.transition2(x_balanced)  # Transition 2

    B, C, H, W = x_balanced.size()
    d_out, d_in = C, H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(
        d_out, d_in, compression_rate_balanced, rank_ratio_balanced
    )

    feat = x_balanced.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    D = torch.clamp(D, min=1e-8)
    normalized = feat / D

    low_rank_comp, sparse_comp = altern_ls(
        normalized, num_iters, target_rank, num_nonzeros
    )
    decomp = (low_rank_comp + sparse_comp) * D / 2
    x_balanced_new = decomp.view(B, C, H, W)

    x_balanced_new = model.features.denseblock3(x_balanced_new)  # Dense Block 3
    x_balanced_new = model.features.transition3(x_balanced_new)  # Transition 3

    x_balanced_final = model.features.denseblock4(x_balanced_new)  # Dense Block 4
    logit_balanced = densenet_logit(model, x_balanced_final)

    x_deep = common_forward(input)
    x_deep = model.features.denseblock1(x_deep)  # Dense Block 1
    x_deep = model.features.transition1(x_deep)  # Transition 1
    x_deep = model.features.denseblock2(x_deep)  # Dense Block 2
    x_deep = model.features.transition2(x_deep)  # Transition 2
    x_deep = model.features.denseblock3(x_deep)  # Dense Block 3
    x_deep = model.features.transition3(x_deep)  # Transition 3

    B, C, H, W = x_deep.size()
    target_rank, num_nonzeros = calculate_lrs_parameters(
        C, H * W, compression_rate_deep, rank_ratio_deep
    )

    feat = x_deep.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    D = torch.clamp(D, min=1e-8)
    normalized = feat / D

    low_rank_comp, _ = altern_ls(
        normalized, num_iters, target_rank, num_nonzeros
    )
    x_deep_new = (low_rank_comp * D).view(B, C, H, W)

    x_deep_final = model.features.denseblock4(x_deep_new)
    logit_lowrank = densenet_logit(model, x_deep_final)

    x_ori = common_forward(input)
    x_ori = model.features.denseblock1(x_ori)
    x_ori = model.features.transition1(x_ori)
    x_ori = model.features.denseblock2(x_ori)
    x_ori = model.features.transition2(x_ori)
    x_ori = model.features.denseblock3(x_ori)
    x_ori = model.features.transition3(x_ori)
    x_ori = model.features.denseblock4(x_ori)
    logit_ori = densenet_logit(model, x_ori)

    return (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4


def multi_lrs_densenet121(model, input, num_iters=5,
                           compression_rate_shallow=0.8, rank_ratio_shallow=0.01,
                           compression_rate_balanced=0.5, rank_ratio_balanced=0.04,
                           compression_rate_deep=0, rank_ratio_deep=0.1):

    return multi_lrs_densenet(
        model, input, num_iters,
        compression_rate_shallow, rank_ratio_shallow,
        compression_rate_balanced, rank_ratio_balanced,
        compression_rate_deep, rank_ratio_deep
    )


def multi_lrs_deit_qkv(model, input, num_iters=5, shallow_layer=4, balanced_layer=7, deep_layer=10,
                        compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                        compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                        compression_rate_deep=0, rank_ratio_deep=0.1):

    x0 = model.patch_embed(input)

    if getattr(model, 'cls_token', None) is not None:
        cls_tokens = model.cls_token.expand(x0.shape[0], -1, -1)
        x0 = torch.cat((cls_tokens, x0), dim=1)

    if hasattr(model, 'dist_token') and model.dist_token is not None:
        dist_token = model.dist_token.expand(x0.shape[0], -1, -1)

        x0 = torch.cat([x0[:, 0:1], dist_token, x0[:, 1:]], dim=1)

    x0 = model.pos_drop(x0 + model.pos_embed)

    def run_blocks(x, k):
        for i in range(k):
            x = model.blocks[i](x)
        return x

    def apply_qkv_decomposition(x, block, compression_rate, rank_ratio):

        x_norm = block.norm1(x)

        qkv = block.attn.qkv(x_norm)

        qkv_decomposed = decompose_qkv_separate(qkv, num_iters, compression_rate, rank_ratio)

        B, N, _ = qkv_decomposed.shape
        C = model.embed_dim
        qkv_reshaped = qkv_decomposed.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
        qkv_reshaped = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]

        attn = (q @ k.transpose(-2, -1)) * block.attn.scale
        attn = attn.softmax(dim=-1)
        attn = block.attn.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = block.attn.proj(x_attn)
        x_attn = block.attn.proj_drop(x_attn)

        x = x + x_attn

        if hasattr(block, 'drop_path'):
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        else:
            x = x + block.mlp(block.norm2(x))

        return x

    # Shallow
    x_shallow = run_blocks(x0.clone(), shallow_layer - 1) if shallow_layer > 0 else x0.clone()
    x_shallow = apply_qkv_decomposition(
        x_shallow, model.blocks[shallow_layer],
        compression_rate_shallow, rank_ratio_shallow
    )
    for i in range(shallow_layer + 1, len(model.blocks)):
        x_shallow = model.blocks[i](x_shallow)
    x_shallow = model.norm(x_shallow)

    logit_shallow = model.head(x_shallow[:, 0])
    del x_shallow

    # Balanced
    x_balanced = run_blocks(x0.clone(), balanced_layer - 1) if balanced_layer > 0 else x0.clone()
    x_balanced = apply_qkv_decomposition(
        x_balanced, model.blocks[balanced_layer],
        compression_rate_balanced, rank_ratio_balanced
    )
    for i in range(balanced_layer + 1, len(model.blocks)):
        x_balanced = model.blocks[i](x_balanced)
    x_balanced = model.norm(x_balanced)
    logit_balanced = model.head(x_balanced[:, 0])
    del x_balanced

    # Deep
    x_deep = run_blocks(x0.clone(), deep_layer - 1) if deep_layer > 0 else x0.clone()
    x_deep = apply_qkv_decomposition(
        x_deep, model.blocks[deep_layer],
        compression_rate_deep, rank_ratio_deep
    )
    for i in range(deep_layer + 1, len(model.blocks)):
        x_deep = model.blocks[i](x_deep)
    x_deep = model.norm(x_deep)
    logit_deep = model.head(x_deep[:, 0])
    del x_deep

    # Original
    x_ori = run_blocks(x0.clone(), len(model.blocks))
    x_ori = model.norm(x_ori)
    logit_ori = model.head(x_ori[:, 0])
    del x_ori

    result = (logit_shallow + logit_balanced + logit_deep + logit_ori) / 4

    del logit_shallow, logit_balanced, logit_deep, logit_ori
    torch.cuda.empty_cache()

    return result



def resnet_logit(model, x):

    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    if hasattr(model, 'fc') and model.fc is not None:
        x = model.fc(x)
    elif hasattr(model, 'last_linear') and model.last_linear is not None:
        x = model.last_linear(x)
    else:
        raise AttributeError("Model does not have 'fc' or 'last_linear' layer")
    return x

def multi_lrs_resnet152(model, input, num_iters=5,
                         compression_rate_shallow=0.8, rank_ratio_shallow=0.01,
                         compression_rate_balanced=0.5, rank_ratio_balanced=0.04,
                         compression_rate_deep=0, rank_ratio_deep=0.1):

    def common_forward(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        return x

    x_shallow = common_forward(input)
    x_shallow = model.layer1(x_shallow)

    B, C, H, W = x_shallow.size()
    d_out, d_in = C, H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(
        d_out, d_in, compression_rate_shallow, rank_ratio_shallow
    )

    feat = x_shallow.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    D = torch.clamp(D, min=1e-8)
    normalized = feat / D

    _, sparse_comp = altern_ls(
        normalized, num_iters, target_rank, num_nonzeros
    )
    x_shallow_new = (sparse_comp * D).view(B, C, H, W)

    x_shallow_new = model.layer2(x_shallow_new)
    x_shallow_new = model.layer3(x_shallow_new)

    logit_shallow = resnet_logit(model, x_shallow_new)

    x_balanced = common_forward(input)
    x_balanced = model.layer1(x_balanced)
    x_balanced = model.layer2(x_balanced)

    B, C, H, W = x_balanced.size()
    d_out, d_in = C, H * W

    target_rank, num_nonzeros = calculate_lrs_parameters(
        d_out, d_in, compression_rate_balanced, rank_ratio_balanced
    )

    feat = x_balanced.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    D = torch.clamp(D, min=1e-8)
    normalized = feat / D

    low_rank_comp, sparse_comp = altern_ls(
        normalized, num_iters, target_rank, num_nonzeros
    )
    decomp = (low_rank_comp + sparse_comp) * D / 2
    x_balanced_new = decomp.view(B, C, H, W)

    x_balanced_new = model.layer3(x_balanced_new)

    logit_balanced = resnet_logit(model, x_balanced_new)

    x_deep = common_forward(input)
    x_deep = model.layer1(x_deep)
    x_deep = model.layer2(x_deep)
    x_deep = model.layer3(x_deep)

    B, C, H, W = x_deep.size()
    target_rank, num_nonzeros = calculate_lrs_parameters(
        C, H * W, compression_rate_deep, rank_ratio_deep
    )

    feat = x_deep.view(B, C, H * W).float()
    D = torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True))
    D = torch.clamp(D, min=1e-8)
    normalized = feat / D

    low_rank_comp, _ = altern_ls(
        normalized, num_iters, target_rank, num_nonzeros
    )
    x_deep_new = (low_rank_comp * D).view(B, C, H, W)

    logit_lowrank = resnet_logit(model, x_deep_new)

    x_ori = common_forward(input)
    x_ori = model.layer1(x_ori)
    x_ori = model.layer2(x_ori)
    x_ori = model.layer3(x_ori)
    logit_ori = resnet_logit(model, x_ori)

    return (logit_shallow + logit_balanced + logit_lowrank + logit_ori) / 4




def multi_lrs_cait_qkv(model, input, num_iters=5, shallow_layer=4, balanced_layer=12, deep_layer=20,
                        compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                        compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                        compression_rate_deep=0, rank_ratio_deep=0.1):

    x0 = model.patch_embed(input)
    x0 = model.pos_drop(x0 + model.pos_embed)

    num_sa_blocks = len(model.blocks)
    num_ca_blocks = len(model.blocks_token_only) if hasattr(model, 'blocks_token_only') else 0

    shallow_layer = min(shallow_layer, num_sa_blocks - 1)
    balanced_layer = min(balanced_layer, num_sa_blocks - 1)
    deep_layer = min(deep_layer, num_sa_blocks - 1)
    
    def run_sa_blocks(x, end_idx):

        for i in range(end_idx):
            x = model.blocks[i](x)
        return x
    
    def apply_qkv_decomposition(x, block, compression_rate, rank_ratio):

        x_norm = block.norm1(x)

        qkv = block.attn.qkv(x_norm)

        qkv_decomposed = decompose_qkv_separate(qkv, num_iters, compression_rate, rank_ratio)

        B, N, C3 = qkv_decomposed.shape
        C = C3 // 3
        qkv_reshaped = qkv_decomposed.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
        qkv_reshaped = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        
        attn = (q @ k.transpose(-2, -1)) * block.attn.scale
        attn = attn.softmax(dim=-1)
        attn = block.attn.attn_drop(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = block.attn.proj(x_attn)
        x_attn = block.attn.proj_drop(x_attn)
        
        x = x + x_attn

        if hasattr(block, 'drop_path'):
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        else:
            x = x + block.mlp(block.norm2(x))
        
        return x
    
    def run_remaining_and_classify(x, start_sa_idx):

        for i in range(start_sa_idx + 1, num_sa_blocks):
            x = model.blocks[i](x)
        
        x = model.norm(x)

        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        

        if num_ca_blocks > 0:
            for ca_block in model.blocks_token_only:
                cls_tokens = ca_block(x, cls_tokens)

        cls_tokens = model.norm(cls_tokens)
        return model.head(cls_tokens[:, 0])
    
    # Shallow
    x_shallow = run_sa_blocks(x0.clone(), shallow_layer)
    x_shallow = apply_qkv_decomposition(
        x_shallow, model.blocks[shallow_layer],
        compression_rate_shallow, rank_ratio_shallow
    )
    logit_shallow = run_remaining_and_classify(x_shallow, shallow_layer)
    del x_shallow
    
    # Balanced
    x_balanced = run_sa_blocks(x0.clone(), balanced_layer)
    x_balanced = apply_qkv_decomposition(
        x_balanced, model.blocks[balanced_layer],
        compression_rate_balanced, rank_ratio_balanced
    )
    logit_balanced = run_remaining_and_classify(x_balanced, balanced_layer)
    del x_balanced
    
    # Deep
    x_deep = run_sa_blocks(x0.clone(), deep_layer)
    x_deep = apply_qkv_decomposition(
        x_deep, model.blocks[deep_layer],
        compression_rate_deep, rank_ratio_deep
    )
    logit_deep = run_remaining_and_classify(x_deep, deep_layer)
    del x_deep
    
    # Original
    logit_ori = model(input)

    result = (logit_shallow + logit_balanced + logit_deep + logit_ori) / 4
    
    del logit_shallow, logit_balanced, logit_deep, logit_ori
    torch.cuda.empty_cache()
    
    return result
