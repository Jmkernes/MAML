import time

class LossMetric:
    """
    Keeps a running average of a scalar value. Probably a better way to do this, but eh, this was faster.
    """
    def __init__(self):
        self.reset()
    def update(self, value):
        self.N += 1
        self.tot += float(value)
    def reset(self):
        self.N, self.tot = 0, 0.0
    def value(self):
        return self.tot/self.N if self.N else 0.0
    def __repr__(self):
        return f"{self.tot/self.N:.04f}"
                            
def print_bar(step, glob_step, ds_size, loss_metric, start, num_epochs, epoch):
    """
    Because I still like the Tensorflow/Keras print bar o.0
    """
    curr_time = time.time()-start
    iter_per_sec = (glob_step) / curr_time
    time_remaining = ((ds_size-step+1)+(num_epochs-epoch)*ds_size)/iter_per_sec
    eq_len = int(20.0*(step+1)/ds_size)
    dot_len = 20-eq_len
    bar = '['+'='*eq_len + '>'+'.'*dot_len+']'
    end = "\r" if step < ds_size-1 else "\n"
    print(f"Epoch: {epoch}. Iter: {glob_step}/{ds_size*num_epochs}  {bar}  Loss: {loss_metric}.\
 Time: {iter_per_sec:.02} iter/s, {time_remaining:.00f} remaining.", end=end)