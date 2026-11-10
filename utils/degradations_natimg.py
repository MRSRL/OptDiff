"""Linear forward operators H for the natural-image tasks (from RED-diff).

Supported degradations (see build_degradation_model):
  srX                                   X-times super-resolution (e.g. sr4, sr16)
  deblur_gauss_window{W}_sigma{S}       Gaussian deblurring with a W-tap kernel of std S
  inpainting_{name}                     inpainting with the mask data/masks/inpainting_{name}.npz
                                        (inpainting_mask: the paper's mask)
  rbox_{B}[_seed{S}]                    random BxB box inpainting
"""
import os
import numpy as np
import torch

MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'masks')


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)  # (b, m) - > (b, m)
        singulars = self.singulars()  # (mxm, )
        # temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]

        return self.V(self.add_zeros(temp))


class Inpainting(H_functions):
    def __init__(self, channels, img_dim, dense_masks, device):
        self.channels = channels
        self.img_dim = img_dim
        self.dense_masks = dense_masks
        n = dense_masks.size(0)
        # print(n)
    
        # self.missing_masks = torch.cat([torch.nonzero(dense_masks[i] == 0).long() for i in range(n)], dim=1).T
        # self.keep_masks = torch.cat([torch.nonzero(dense_masks[i] != 0).long() for i in range(n)], dim=1).T
        self.missing_masks = []
        self.keep_masks = []

        for i in range(n):
            mask = dense_masks[i].view(-1)  # flatten (3*H*W)

            missing = torch.nonzero(mask == 0, as_tuple=False).squeeze(1)
            kept    = torch.nonzero(mask != 0, as_tuple=False).squeeze(1)

            self.missing_masks.append(missing)
            self.keep_masks.append(kept)

        self.missing_masks = torch.stack(self.missing_masks)  # (n, num_missing)
        self.keep_masks    = torch.stack(self.keep_masks)     # (n, num_kept)
        self.device = device
        
        
    def set_indices(self, idx):

        device = self.device
        idx = torch.remainder(idx, self.missing_masks.size(0))
        missing_masks = self.missing_masks[idx].clone().to(device)

        missing_masks = missing_masks
        # print('missing_masks.shape', missing_masks.shape)
        self.missing_indices = missing_masks #.unsqueeze(0) # self.missing_masks[idx].clone().to(device)
        # self._singulars = torch.ones(channels * img_dim ** 2 - missing_indices.shape[1]).to(device)
        kept_masks = self.keep_masks[idx].clone().to(device)
        self.kept_indices = kept_masks#.unsqueeze(0) # self.keep_masks[idx].clone().to(device)
        self._singulars = torch.ones(self.kept_indices.shape[0], device=device)
        # print("kept:", self.kept_indices.shape)
        # print("missing:", self.missing_indices.shape)
        # print("total:", self.channels * self.img_dim ** 2)
        

    def V(self, vec):
        # temp = vec.clone().reshape(vec.shape[0], -1)
        # out = torch.zeros_like(temp)
        # assert vec.size(0) == self.kept_indices.size(0)
        # n = vec.size(0)
        # for i in range(n):
        #     out[i, self.kept_indices[i]] = temp[i, : self.kept_indices[i].shape[0]]
        #     out[i, self.missing_indices[i]] = temp[i, self.kept_indices[i].shape[0] :]
        # return out #.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)
        B = vec.shape[0]
        # print(vec.shape)
        vec = vec.view(B, -1)
        out = torch.zeros(B, self.channels * self.img_dim ** 2, device=vec.device)
        out[:, self.kept_indices] = vec[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = vec[:, self.kept_indices.shape[0]:]
        return out
    
    def Vt(self, vec):
        # temp = vec.clone().reshape(vec.shape[0], -1)
        # out = torch.zeros_like(temp)
        # assert vec.size(0) == self.kept_indices.size(0)
        # n = vec.size(0)
        # for i in range(n):
        #     out[i, : self.kept_indices[i].shape[0]] = temp[i, self.kept_indices[i]]
        #     out[i, self.kept_indices[i].shape[0] :] = temp[i, self.missing_indices[i]]
        # return out
        B = vec.shape[0]
        vec = vec.view(B, -1)
        # print(vec.shape)
        out = torch.zeros(B, self.channels * self.img_dim ** 2, device=vec.device)
        out[:, :self.kept_indices.shape[0]] = vec[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = vec[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        # temp = self.Vt(vec)
        # singulars = self.singulars()
        # return self.U(singulars * temp[:, :singulars.shape[0]])
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars[None, :] * temp[:, :singulars.shape[0]])


    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        # temp = self.Ut(vec)
        # singulars = self.singulars()
        # return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars[None, :] * temp[:, :singulars.shape[0]]))


    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        # temp = self.Ut(vec)  # (b, m) - > (b, m)
        # singulars = self.singulars()  # (mxm, )
        # # temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        # nonzero_idx = singulars.nonzero().flatten()
        # temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]

        # return self.V(self.add_zeros(temp))
        temp = self.Ut(vec)
        singulars = self.singulars()
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] /= singulars[nonzero_idx][None, :]
        return self.V(self.add_zeros(temp))


class RandomBoxInpainting(Inpainting):
    """
    Same masking math as `Inpainting`, but instead of indexing into a
    precomputed table of masks, the box mask for a given dataset index is
    generated on the fly from `seed + index`. This keeps the mapping from
    image index -> box location fixed across separate runs (e.g. this method
    vs. a baseline), which is what you need for a fair per-image comparison,
    without having to precompute/store one mask per dataset image.

    Only a single scalar index per call is supported (i.e. one image per
    `H()` call) — `Inpainting`'s V/Vt only broadcast correctly when the same
    mask is shared across the whole batch, so genuinely different boxes for
    different images within one batched forward pass aren't supported here.
    """

    def __init__(self, channels, img_dim, box_size, seed, device):
        assert box_size <= img_dim, f"box_size {box_size} must be <= img_dim {img_dim}"
        self.channels = channels
        self.img_dim = img_dim
        self.box_size = box_size
        self.seed = seed
        self.device = device

    def _box_mask(self, index):
        rng = np.random.RandomState(self.seed + int(index))
        max_start = self.img_dim - self.box_size
        top = rng.randint(0, max_start + 1)
        left = rng.randint(0, max_start + 1)
        mask = torch.ones(self.img_dim, self.img_dim, dtype=torch.bool)
        mask[top:top + self.box_size, left:left + self.box_size] = False
        return mask.unsqueeze(0).expand(self.channels, -1, -1).reshape(-1)

    def set_indices(self, idx):
        idx = torch.as_tensor(idx)
        if idx.dim() != 0:
            raise NotImplementedError(
                "RandomBoxInpainting only supports a single scalar index per "
                "H() call (one image per forward pass)."
            )
        mask = self._box_mask(idx.item())
        self.missing_indices = torch.nonzero(mask == 0, as_tuple=False).squeeze(1).to(self.device)
        self.kept_indices = torch.nonzero(mask != 0, as_tuple=False).squeeze(1).to(self.device)
        self._singulars = torch.ones(self.kept_indices.shape[0], device=self.device)


class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim)).reshape(
            v.shape[0], self.channels, M.shape[0], self.img_dim
        )

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim), M).reshape(
            v.shape[0], self.channels, self.img_dim, M.shape[1]
        )

    def __init__(self, kernel, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                if j < 0 or j >= img_dim:
                    continue
                H_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim),
        ).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio ** 2] * ratio ** 2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly): the first
        # channels * y_dim^2 entries are the patches' first components, the rest holds the other ratio^2 - 1
        # components of each patch in turn
        temp = vec.reshape(vec.shape[0], -1)
        n = self.channels * self.y_dim ** 2
        patches = torch.cat([temp[:, :n].reshape(vec.shape[0], self.channels, -1, 1),
                             temp[:, n:].reshape(vec.shape[0], self.channels, -1, self.ratio ** 2 - 1)], dim=3)
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio ** 2
        )
        # repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio ** 2
        )
        # reorder the vector to have the first entry first (because singulars are ordered descendingly): the
        # patches' first components, then the other ratio^2 - 1 components of each patch in turn (inverse of V)
        recon = torch.cat([patches[:, :, :, 0].reshape(vec.shape[0], -1),
                           patches[:, :, :, 1:].reshape(vec.shape[0], -1)], dim=1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


def pdf(x, sigma=10):
    return torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2]))


def load_mask(f):
    d = np.load(f)
    m = d['m']
    shape = d['shape']
    m = np.unpackbits(m, count=np.prod(shape)).reshape(shape).view(bool)
    return m

def build_degradation_model(deg, image_size=256, channels=3, device="cpu"):
    print('deg', deg)
    return build_one_degradation_model(image_size, image_size, channels, deg, device)


def build_one_degradation_model(h, w, c, deg: str, device):
    if deg.startswith("inpainting_"):  # the degradation names its mask file
        masks_root = os.path.join(MASKS_DIR, f'{deg}.npz')
        m = load_mask(masks_root)
        
        if h!=256 and w!=256:
            n = m.shape[0]
            m = np.reshape(m, (n, 3, 256, 256))
            m = np.repeat(m, w//256, axis=2)
            m = np.repeat(m, h//256, axis=3)
            m = np.reshape(m, (n, 3*h*w))
            # print('m.shape', m.shape)
            # print('m', m)
            
        m = torch.from_numpy(m).unsqueeze(0) # adding batch dim
        H = Inpainting(c, w, m, device)
        # H = Inpainting2(c, w, m, device)
    elif deg[:4] == "rbox":
        # format: "rbox_<box_size>" or "rbox_<box_size>_seed<seed>"
        # e.g. "rbox_64" -> 64x64 box, default seed 0
        #      "rbox_64_seed7" -> 64x64 box, seed 7
        parts = deg.split('_')
        box_size = int(parts[1])
        seed = 0
        for p in parts[2:]:
            if p.startswith('seed'):
                seed = int(p[len('seed'):])
        H = RandomBoxInpainting(c, w, box_size, seed, device)
    elif "deblur_gauss_window" in deg: # gauss_deblur_window{}_sigma{}
        parts = deg.split('_')
        window = int(parts[2].replace('window', ''))
        sigma = float(parts[3].replace('sigma', ''))
        kernel = [pdf(t, sigma) for t in range(-(window-1)//2, (window-1)//2)]
        kernel = torch.Tensor(kernel).to(device)
        H = Deblurring(kernel / kernel.sum(), c, w, device)
    elif deg[:2] == "sr":
        downsample_by = int(deg[2:])
        H = SuperResolution(c, w, downsample_by, device)
    else:
        raise ValueError(f"Degradation model {deg} does not exist.")

    return H
