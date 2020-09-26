from .model import *
from .m_loss import HuberLoss

class m_model(model):
    def __init__(self, loss: str="huber", lambda1: float=0.0, lambda2: float=0.0):
        super().__init__("M estimator", lambda1, lambda2)
        self.__loss = loss
        if self.__loss.lower() == 'huber':
            self.__l = 3.0 # default for huber loss
            self.__lossobj = HuberLoss()
        else:
            print("At the moment the module only supports huber loss. Set loss='huber'.")

    @property
    def loss(self): return self.__loss

    @property
    def l(self): return self.__l

    def gradient(self, t:int, theta_old: np.ndarray, data: data_set) -> np.ndarray:
        datum = data.get_data_point(t)
        return self.__lossobj.first_deriv(datum._y - (datum._x @ theta_old), self.__l) * datum._x - self.gradient_penalty(theta_old)

    ## Functions for the implicit iteration
    def scale_factor(self, ksi: float, at: float, datum: data_point, theta_old: np.ndarray, normx: float) -> float:
        return self.__lossobj.first_deriv(datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx, self.__l)

    def scale_factor_first_deriv(self, ksi: float, at: float, datum: data_point, theta_old: np.ndarray, normx: float) -> float:
        return self.__lossobj.second_deriv(datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx, self.__l) * normx

    def scale_factor_second_deriv(self, ksi: float, at: float, datum: data_point, theta_old: np.ndarray, normx: float) -> float:
        return self.__lossobj.third_deriv(datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx, self.__l) * normx * normx
