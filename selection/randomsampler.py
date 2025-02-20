from selection.selectivesampler import SelectiveSampler
import random


class RandomSampler(SelectiveSampler):
    """Example implementation of RandomSampler that chooses a random amount of samples."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
       super().__init__(
           dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
       )
       self.mask = None
       self.set_num_selected_samples()
       
       

    def set_num_selected_samples(self):
        """Set expected number of selected samples for dataset len"""
        choice_amount = random.randint(1,len(self.dataset))
        self._num_selected_samples = choice_amount


    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        n = len(self.dataset)
        mask = [False] * n
        mask[:self._num_selected_samples] = [True] * self._num_selected_samples
        self.set_mask(mask)

    def post_epoch(self) -> None:
        """No-op post epoch hook"""
        pass

    def on_batch(self, idx: int, batch: dict) -> None:
        """No-op batch hook"""
        pass

    def after_forward(self, idx: int, batch: dict, current_loss: float) -> None:
        """No-op after forward hook"""
        pass
    
    

