import torch
import torch.nn as nn

from scipy.stats import norm
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class Smooth(nn.Module):

    ABSTAIN = -1  # to abstain, Smooth returns this int

    def __init__(
        self, base_classifier: torch.nn.Module, 
        sigma: float, N: int, alpha: float, batch_size=None, num_classes=10
    ):
        """
        :param base_classifier: maps from [batch x channel x height x width] 
                                to [batch x num_classes]
        :param num_classes:
        :param sigma:           the noise level hyperparameter
        """
        super().__init__()
        
        self.base_classifier = base_classifier
        self.sigma, self.N, self.alpha = sigma, N, alpha
        self.batch_size = self.N if batch_size is None else batch_size
        self.num_classes = num_classes
        self.mode = "predict"  # "certify"

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):
        #######   THIS FUNCTION HAS NOT BEEN TESTED WITH ADAPTIVE SMOOTHING   #######

        """ Monte Carlo algorithm for certifying that g's prediction 
            around x is constant within some L2 radius.
        With probability at least 1 - alpha, 
        the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x:           the input [channel x height x width]
        :param n0:          the number of Monte Carlo samples to use for selection
        :param n:           the number of Monte Carlo samples to use for estimation
        :param alpha:       the failure probability
        :param batch_size:  batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
            in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size).cpu().numpy()
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size).cpu().numpy()
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return self.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor) -> torch.tensor:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  
        With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        Uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x:           the input [num_imgs x channel x height x width]
        :param n:           the number of Monte Carlo samples to use
        :param alpha:       the failure probability
        :param batch_size:  batch size to use when evaluating the base classifier
        :return:            the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, self.N, self.batch_size)
        return counts

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> torch.tensor:
        """ Sample the base classifier's prediction
            under noisy corruptions of the input x.

        :param x:           The input [num_imgs x channel x width x height]
        :param num:         Number of samples to collect
        :param batch_size:  Batch size.
        :return:            A Tensor[int] of length num_classes containing
                            the per-class counts
        """
        with torch.no_grad():
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
                single = True
            else:
                single = False
            n, _, _, _ = x.shape

            counts = torch.zeros((n, self.num_classes), dtype=int).cuda()
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                imgs_with_noise = (batch + noise).flatten(0, 1)
                predictions = self.base_classifier(imgs_with_noise)[0].argmax(1)
                predictions = predictions.unflatten(0, (this_batch_size, n))

                counts = counts + self._count_arr(predictions, self.num_classes)
            return counts.squeeze(0) if single else counts

    def _count_arr(self, arr: torch.tensor, length: int) -> torch.tensor:
        """bincount along the 0th direction for each distinct image
        :param arr: this_batch_size x num_imgs
        :return:    num_imgs x num_classes
        """
        counts = torch.zeros((arr.shape[1], length), dtype=int).cuda()
        for ind in range(arr.shape[1]):
            counts[ind, :] = arr[:, ind].bincount(minlength=10)
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA:      the number of "successes"
        :param N:       the number of total draws
        :param alpha:   the confidence level
        :return:        a lower bound on the binomial proportion which holds true 
                        w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    def forward(self, x: torch.tensor):
        if self.mode == "predict":
            return self.predict(x), None
        else:
            return self.certify(x)  # Not tested
