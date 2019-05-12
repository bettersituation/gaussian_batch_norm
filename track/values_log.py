import numpy as np


class ValuesLog:
    def __init__(self, normed_keys, gradients_keys):
        self.normed_keys = normed_keys
        self.gradient_keys = gradients_keys

        self.batch_reg_losses = list()
        self.batch_losses = list()
        self.batch_accs = list()

        self.normed_batch_maxs = {k: list() for k in normed_keys}
        self.normed_batch_mins = {k: list() for k in normed_keys}
        self.normed_epoch_max = {k: None for k in normed_keys}
        self.normed_epoch_min = {k: None for k in normed_keys}
        
        self.gradient_batch_maxs = {k: list() for k in gradients_keys}
        self.gradient_batch_mins = {k: list() for k in gradients_keys}
        self.gradient_batch_norms = {k: list() for k in gradients_keys}
        self.gradient_epoch_max = {k: None for k in gradients_keys}
        self.gradient_epoch_min = {k: None for k in gradients_keys}
        self.gradient_epoch_norm = {k: None for k in gradients_keys}

    def save_batchs(self, reg_loss, loss, acc, normed_batch_max, normed_batch_min, gradient_batch_max=None, gradient_batch_min=None, gradient_batch_norm=None):
        self.batch_reg_losses.append(reg_loss)
        self.batch_losses.append(loss)
        self.batch_accs.append(acc)

        self.save_normed_batch_max(normed_batch_max)
        self.save_normed_batch_min(normed_batch_min)
        if gradient_batch_max is not None:
            self.save_gradient_batch_max(gradient_batch_max)
        if gradient_batch_min is not None:
            self.save_gradient_batch_min(gradient_batch_min)
        if gradient_batch_norm is not None:
            self.save_gradient_batch_norm(gradient_batch_norm)
        
    def save_normed_batch_max(self, normed_batch_max):
        for k, v in normed_batch_max.items():
            self.normed_batch_maxs[k].append(v)
        
    def save_normed_batch_min(self, normed_batch_min):
        for k, v in normed_batch_min.items():
            self.normed_batch_mins[k].append(v)
        
    def save_gradient_batch_max(self, gradient_batch_max):
        for k, v in gradient_batch_max.items():
            self.gradient_batch_maxs[k].append(v)
        
    def save_gradient_batch_min(self, gradient_batch_min):
        for k, v in gradient_batch_min.items():
            self.gradient_batch_mins[k].append(v)

    def save_gradient_batch_norm(self, gradient_batch_norm):
        for k, v in gradient_batch_norm.items():
            self.gradient_batch_norms[k].append(v)

    def clear_batch_and_epochs(self):
        self.batch_reg_losses.clear()
        self.batch_losses.clear()
        self.batch_accs.clear()

        for k in self.normed_keys:
            self.normed_batch_maxs[k].clear()
            self.normed_batch_mins[k].clear()
            self.normed_epoch_max[k] = None
            self.normed_epoch_min[k] = None
        
        for k in self.gradient_keys:
            self.gradient_batch_maxs[k].clear()
            self.gradient_batch_mins[k].clear()
            self.gradient_batch_norms[k].clear()
            self.gradient_epoch_max[k] = None
            self.gradient_epoch_min[k] = None
            self.gradient_epoch_norm[k] = None
            
    def calc_batchs(self):
        for k in self.normed_keys:
            self.normed_epoch_max[k] = max(self.normed_batch_maxs[k])
            self.normed_epoch_min[k] = min(self.normed_batch_mins[k])
            
        for k in self.gradient_keys:
            if len(self.gradient_batch_maxs[k]) != 0:
                self.gradient_epoch_max[k] = max(self.gradient_batch_maxs[k])
            else:
                self.gradient_epoch_max[k] = None

            if len(self.gradient_batch_mins[k]) != 0:
                self.gradient_epoch_min[k] = min(self.gradient_batch_mins[k])
            else:
                self.gradient_epoch_min[k] = None

            if len(self.gradient_batch_norms[k]) != 0:
                self.gradient_epoch_norm[k] = sum(self.gradient_batch_norms[k]) / len((self.gradient_batch_norms[k]))
            else:
                self.gradient_epoch_norm[k] = None

    def get_epochs(self):
        reg_loss = sum(self.batch_reg_losses) / len(self.batch_reg_losses)
        loss = sum(self.batch_losses) / len(self.batch_losses)
        acc = sum(self.batch_accs) / len(self.batch_accs)

        normed_max = self.normed_epoch_max.copy()
        normed_min = self.normed_epoch_min.copy()
        gradient_max = self.gradient_epoch_max.copy()
        gradient_min = self.gradient_epoch_min.copy()
        gradient_norm = self.gradient_epoch_norm.copy()
        return reg_loss, loss, acc, normed_max, normed_min, gradient_max, gradient_min, gradient_norm

    def get_global_epochs(self):
        normed_max_of_max = self._get_global_max(self.normed_epoch_max)
        normed_min_of_min = self._get_global_min(self.normed_epoch_min)
        grad_max_of_max = self._get_global_max(self.gradient_epoch_max)
        grad_min_of_min = self._get_global_min(self.gradient_epoch_min)
        return normed_max_of_max, normed_min_of_min, grad_max_of_max, grad_min_of_min

    @staticmethod
    def _get_global_max(values_dict):
        max_v = - np.inf
        for v in values_dict.values():
            if v is None:
                return None
            if v > max_v:
                max_v = v
        return max_v
    
    @staticmethod
    def _get_global_min(values_dict):
        min_v = np.inf
        for v in values_dict.values():
            if v is None:
                return None

            if v < min_v:
                min_v = v
        return min_v
