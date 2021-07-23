import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy

class LossImportance:
    """
    Creates a loss weighting vector (inverse discount) to discount earlier training steps.
    
    Attributes:
        K: number of training steps
        decay_rate: fixing the decay rate. losses are discounted as e^{-range(K) * decay_rate}
        train_steps: Makes sure the 2nd to last loss reaches a weight of 1e-2 after train_steps steps.
    """
    def __init__(self, K, decay_rate=None, train_steps=None):
        if decay_rate is None and train_steps is None:
            self.decay_rate = -2e-3 # This is a pretty good default, no special reason here.
        elif decay_rate is None:
            self.decay_rate = np.log(1e-2) / (train_steps+1) # small calculation to get full decay after train_steps
        else:
            self.decay_rate = decay_rate
        self.K = K
        
    def __call__(self, glob_step):
        return torch.tensor(np.exp(self.decay_rate * np.arange(self.K-1, -1, -1) * glob_step),
                            requires_grad=False).float()
    
def parse_attr_name(obj, name_string):
    """
    This is necessary due to how Sequential models are saved, they get numbers.
    
    Args:
        obj: a pytorch model
        name_string: the full attribute name given in string form, like "model.linear.3.dropout"
        
    Returns:
        obj: the object furthest down. In the above example it'd be "dropout".
        name_string: the name of that object.
    """
    names = name_string.split('.')
    for name in names[:-1]:
        obj = obj[int(name)] if name.isdigit() else getattr(obj, name)
        
    return obj, names[-1]

def set_model_attrs(model, param_dict):
    """
    This is pure hackery. It manually replaces all of the model weights with those in param_dict
    
    Args:
        model: a pytorch model
        param_dict: weights to attach to the model architecture.
        
    Returns:
        Nothing. It modifies state.
    """
    for name, param in param_dict.items():
        obj, name = parse_attr_name(model, name)        
        delattr(obj, name)
        setattr(obj, name, param)

def meta_forward(model, x, y, loss_fn, loss_importance, trainable_lrs, K, create_graph=True):
    """
    Runs K train steps on a particular batch, and returns the meta loss and trained parameter dict.
    
    Args:
        model: pytorch model
        x: a batch of model inputs for a single particular class.
        y: a batch of model labels for a single particular class.
        loss_importance: a K dimensional vector, giving weights assigned to the loss output of all K train steps.
        trainable_lrs: nn.ParameterDict() containing the learnable learning rates for each weight.
        K: number of train steps per batch.
        create_graph: a boolean with True retaining the graph for 2nd order derivatives.
        
    Returns:
        Tuple (loss, param_dict). Loss is the meta-loss, of the trained model on a batch of some class j,
        on a test batch of class j. param_dict
    """
    if loss_importance is None:
        loss_importance = torch.zeros(K)
    
    # rip out the model architecture, and replace the weights with our original model weights
    skeleton = deepcopy(model)
    param_dict = OrderedDict(model.named_parameters())
    set_model_attrs(skeleton, param_dict)
    
    # this is a MAML++ step. we accumulate losses over each training iteration
    batch_loss = []
    
    for i in range(K):
        loss = loss_fn(skeleton(x), y)
        
        if i < K-1:
            
            # create_graph=True keeps the graph, enabling 2nd order derivatives during the test step
            grads = torch.autograd.grad(loss, param_dict.values(), create_graph=create_graph)
            
            for (name, param), grad in zip(param_dict.items(), grads):
                lr = F.relu(trainable_lrs[name.replace('.','-')][i])
                param_dict[name] = param - lr * grad
                
            set_model_attrs(skeleton, param_dict)
        batch_loss.append(loss)
        
    # loss_importance is like an inverse discount from RL. A vector like [0.001, 0.01, 0.1, 1]
    # that over time tends to [0, 0, 0, 1] so that only the outcome of the final training state counts
    loss = torch.sum(torch.stack(batch_loss) * loss_importance)
    return loss, param_dict

def meta_adapt(model, x, y, loss_fn):
    """
    Runs the training phase on input and targets x and y. Returns the trained model.
    """
    _, param_dict = meta_forward(model, x, y, loss_fn, loss_importance=None, create_graph=False)
    model = deepcopy(model)
    model.load_state_dict(OrderedDict({k:param_dict[k] for k,_ in model.state_dict()}))
    return model