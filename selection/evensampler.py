from selection.selectivesampler import SelectiveSampler
import random


class EvenSampler(SelectiveSampler):
    """Example implementation of EvenSampler that chooses only samples even indexes."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
       super().__init__(
           dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
       )
       self.mask = None
       self.set_num_selected_samples()
       
       

    def set_num_selected_samples(self):
        """Set expected number of selected samples for dataset len"""
        n = (self.num_samples + 1) // 2
        self._num_selected_samples = n 


    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        mask = []
        for i in range(0, self.num_samples) : 
            if i % 2 != 0 : 
                mask.append(False)
            
            else :
                mask.append(True)
                
        self.set_mask(mask)
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
