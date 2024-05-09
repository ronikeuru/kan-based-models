import torch

def b_splines(
        x: torch.Tensor,
        grid: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
    """Generate B-splines.

        Args:
            x (Tensor): [batch_size, in_dim]
            grid (Tensor): [in_dim, grid_size + 2*k + 1]
            k: (int): degree of B-splines

        Returns:
            B (Tensor): [batch_size, in_dim, grid_size + k]
    """
    x = x.unsqueeze(-1)

    B = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    for p in range(1, k + 1):
        """B-Splines formula
        https://en.wikipedia.org/wiki/B-spline

                    (x   - x_0)            (x_3 - x  )
            B(x) := ----------- * B_prev + ----------- * B_next
                    (x_1 - x_0)            (x_3 - x_2)
        """

        B_prev = B[:, :, :-1]
        B_next = B[:, :, 1:]

        x_0 = grid[:, :-(p + 1)]
        x_1 = grid[:, p:-1]
        coeff_1 = (x - x_0) / (x_1 - x_0)

        x_2 = grid[:, 1:-p]
        x_3 = grid[:, p + 1:]
        coeff_2 = (x_3 - x) / (x_3 - x_2)

        B = coeff_1 * B_prev + coeff_2 * B_next
    
    return B