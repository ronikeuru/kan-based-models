import torch

from .splines import b_splines

def curve2coeff(
        x: torch.Tensor,
        y: torch.Tensor,
        grid: torch.Tensor,
        k: int,
) -> torch.Tensor:
    """Calculate coefficients of B-splines
        Args:
            x (Tensor): [batch_size, in_dim]
            y (Tensor): [batch_size, in_dim, out_dim]
            grid (Tensor): [in_dim, grid_size + 2*k + 1]
            k (int): degree of B-splines
        
        Returns:
            coeff (Tensor): [out_dim, in_dim, grid_size + k]
    """
    splines = b_splines(x, grid, k).transpose(0, 1)
    device = splines.device
    if device != "cpu":
        splines = splines.cpu()
        y = y.cpu()
    
    coeff = torch.linalg.lstsq(splines, y.transpose(0, 1)).solution.to(device).permute(2, 0, 1)

    return coeff