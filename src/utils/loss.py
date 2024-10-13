import torch
import torch.nn as nn

class CoxPHLoss(nn.Module):
    """
    Implements the Cox Proportional Hazard Loss 
    for Time-to-Event prediction in the presence of censored data.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        pred: torch.Tensor, 
        time: torch.Tensor,
        event: torch.Tensor 
        ):

        """
        Expects pred event and time to have the same shape.
        """

        order = torch.argsort(time, descending=True)
        pred = pred[order]
        event = event[order]

        risk_set_log = torch.logcumsumexp(pred, dim=0)
        anchor_loss = pred - risk_set_log
        anchor_loss = anchor_loss[event == 1]
        
        N = event.sum()
        loss = -anchor_loss.sum() / N

        return loss
    


class RankDeepSurvLoss(nn.Module):
    """
    Implements Rank Deep Surv Loss from 
    https://www.sciencedirect.com/science/article/abs/pii/S0933365718305992
    """
    def __init__(
        self, 
        alpha: float = 1.0, 
        beta: float = 1.0
        ):
        super(RankDeepSurvLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction="none")

    def _l1_loss(self, pred, time, event):
        mse = self.mse(pred, time)
        mask = (event == 1) | ((event == 0) & (pred <= time))
        l1_loss = mse[mask].sum() / event.shape[0]

        return l1_loss
    
    def _l2_loss(self, pred, time, event):
        time_diff = time.unsqueeze(1) - time.unsqueeze(0)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)

        event_i = event.unsqueeze(1)
        event_j = event.unsqueeze(0)

        comp_matrix = (event_i == 1) & ((event_j == 1) | (((event_i == 1) & (event_j == 0)) & (time_diff <= 0)))

        mse = self.mse(time_diff, pred_diff)

        l2_loss = mse[comp_matrix].sum() / event.shape[0]

        return l2_loss

    def forward(self, pred, time, event):
        l1_loss = self._l1_loss(pred, time, event)
        l2_loss = self._l2_loss(pred, time, event)

        loss = (self.alpha * l1_loss) + (self.beta * l2_loss)

        return loss