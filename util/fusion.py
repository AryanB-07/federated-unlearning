import abc
import torch
from torch import nn
import copy

""""
Fusion: base class for fusion algorithms
FusionAvg: compute average across all parties
FusionRetrain: compute average across all parties except the target one
"""


class Fusion(abc.ABC):
    """
    Base class for Fusion
    """
    def __init__(self, num_parties, party_sample_counts=None):
        self.name = "fusion"
        self.num_parties = num_parties
        self.party_sample_counts = party_sample_counts  # NEW: track dataset sizes
        
    def average_selected_models(self, selected_parties, party_models):
        with torch.no_grad():
            # Calculate weights based on dataset sizes
            if self.party_sample_counts is not None:
                # Weighted averaging
                total_samples = sum([self.party_sample_counts[i] for i in selected_parties])
                weights = [self.party_sample_counts[i] / total_samples for i in selected_parties]
            else:
                # Fallback to uniform averaging
                weights = [1.0 / len(selected_parties)] * len(selected_parties)
            
            # Weighted sum
            sum_vec = weights[0] * nn.utils.parameters_to_vector(party_models[selected_parties[0]].parameters())
            for i in range(1, len(selected_parties)):
                sum_vec += weights[i] * nn.utils.parameters_to_vector(party_models[selected_parties[i]].parameters())
            
            model = copy.deepcopy(party_models[0])
            nn.utils.vector_to_parameters(sum_vec, model.parameters())
        return model.state_dict()
            
    @abc.abstractmethod
    def fusion_algo(self, party_models, current_model=None):
        raise NotImplementedError


class FusionAvg(Fusion):
    def __init__(self, num_parties, party_sample_counts=None):
        super().__init__(num_parties, party_sample_counts)
        self.name = "Fusion-Average"
    
    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict


class FusionRetrain(Fusion):
    def __init__(self, num_parties, party_sample_counts=None):
        super().__init__(num_parties, party_sample_counts)
        self.name = "Fusion-Retrain"
        
    # Currently, we assume that the party to be erased is party_id = 0
    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(1, self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict
