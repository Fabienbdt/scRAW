"""
Algorithm Registry for extensible algorithm management.
Allows easy registration and discovery of new algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import logging

from .config import HyperparameterConfig

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmInfo:
    """Information about a registered algorithm."""
    name: str
    display_name: str
    description: str
    category: str  # 'deep_learning', 'classical', 'graph_based', etc.
    is_graph_based: bool = False
    requires_gpu: bool = False
    supports_labels: bool = True
    supports_out_of_sample: bool = True
    # Preprocessing guidance for users
    preprocessing_notes: str = ""  # Displayed in UI to guide user
    has_internal_preprocessing: bool = False  # True if algo does its own normalization/transformation
    recommended_data: str = "preprocessed"  # "raw", "preprocessed", or "either"
    # MPS (Apple Silicon) compatibility
    mps_compatible: bool = True  # True if algorithm works with MPS
    mps_notes: str = ""  # Notes about MPS limitations if any



class BaseAlgorithm(ABC):
    """
    Base class for all clustering algorithms.
    Inherit from this class to create new algorithms.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the algorithm with parameters.

        Args:
            params: Dictionary of hyperparameters
        """
        self.params = params or {}
        # Algorithms can publish runtime-resolved params (after fallback/clamping)
        # to improve reproducibility artifacts.
        self._effective_params: Dict[str, Any] = dict(self.params)
        self._fitted = False
        self._labels = None
        self._embeddings = None
        self._loss_history: List[Dict[str, Any]] = []
        self._embedding_snapshots: List[Dict[str, Any]] = []

    @classmethod
    @abstractmethod
    def get_info(cls) -> AlgorithmInfo:
        """
        Return information about this algorithm.

        Returns:
            AlgorithmInfo object describing this algorithm
        """
        pass

    @classmethod
    @abstractmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        """
        Return the list of hyperparameters for this algorithm.

        Returns:
            List of HyperparameterConfig objects
        """
        pass

    @abstractmethod
    def fit(self, data: Any, labels: Optional[Any] = None) -> 'BaseAlgorithm':
        """
        Fit the algorithm to the data.

        Args:
            data: Input data (typically AnnData object)
            labels: Optional ground truth labels

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels.

        Args:
            data: Optional new data to predict (uses fitted data if None)

        Returns:
            Predicted cluster labels
        """
        pass

    def fit_predict(self, data: Any, labels: Optional[Any] = None) -> Any:
        """
        Fit and predict in one step.

        Args:
            data: Input data
            labels: Optional ground truth labels

        Returns:
            Predicted cluster labels
        """
        self.fit(data, labels)
        return self.predict()

    def get_embeddings(self) -> Optional[Any]:
        """
        Get the latent embeddings if available.

        Returns:
            Latent embeddings or None if not available
        """
        return self._embeddings

    def get_labels(self) -> Optional[Any]:
        """
        Get the predicted labels.

        Returns:
            Predicted labels or None if not fitted
        """
        return self._labels
    
    def get_num_parameters(self) -> Optional[int]:
        """
        Get the number of trainable parameters in the model.

        Returns:
            Number of parameters or None if not applicable/implemented
        """
        return None

    @staticmethod
    def _coerce_float_list(values: Any) -> List[float]:
        """Normalize scalar/array-like values to a flat float list."""
        if values is None:
            return []

        if isinstance(values, list):
            raw_list = values
        elif isinstance(values, (tuple, range)):
            raw_list = list(values)
        elif hasattr(values, "tolist"):
            try:
                raw_list = values.tolist()
            except Exception:
                raw_list = [values]
        else:
            raw_list = [values]

        if raw_list and isinstance(raw_list[0], (list, tuple)):
            flat: List[Any] = []
            for item in raw_list:
                if isinstance(item, (list, tuple)):
                    if len(item) == 1:
                        flat.append(item[0])
                    else:
                        flat.extend(item)
                else:
                    flat.append(item)
            raw_list = flat

        out: List[float] = []
        for item in raw_list:
            if item is None:
                continue
            try:
                if hasattr(item, "item"):
                    item = item.item()
            except Exception:
                pass
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _coerce_epoch_list(values: Any, default_len: int) -> List[int]:
        """Normalize epoch values; fallback to range(default_len) when invalid."""
        if values is None:
            return list(range(default_len))

        if isinstance(values, list):
            raw_list = values
        elif isinstance(values, (tuple, range)):
            raw_list = list(values)
        elif hasattr(values, "tolist"):
            try:
                raw_list = values.tolist()
            except Exception:
                raw_list = []
        else:
            raw_list = []

        out: List[int] = []
        for item in raw_list:
            if item is None:
                continue
            try:
                if hasattr(item, "item"):
                    item = item.item()
            except Exception:
                pass
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue

        if not out:
            return list(range(default_len))
        return out

    def get_loss_history(self) -> List[Dict[str, Any]]:
        """
        Get the training loss history.

        Returns:
            List of phase dicts, each with keys:
              - name: phase name (e.g. 'pretrain', 'clustering')
              - epochs: list of epoch indices
              - train_loss: list of loss values per epoch
              - val_loss: optional list of validation loss values per epoch
              - components: dict mapping component name to list of values (optional)
        """
        raw_history = self._loss_history if isinstance(self._loss_history, list) else []
        normalized_history: List[Dict[str, Any]] = []

        for idx, raw_phase in enumerate(raw_history):
            if not isinstance(raw_phase, dict):
                continue

            phase = dict(raw_phase)
            train_loss = self._coerce_float_list(phase.get('train_loss', []))
            val_loss = self._coerce_float_list(phase.get('val_loss', []))
            max_len = max(len(train_loss), len(val_loss))

            epochs = self._coerce_epoch_list(phase.get('epochs'), default_len=max_len)
            if max_len > 0 and len(epochs) != max_len:
                epochs = list(range(max_len))

            phase['name'] = str(phase.get('name', f'Phase {idx + 1}'))
            phase['epochs'] = epochs
            phase['train_loss'] = train_loss
            phase['val_loss'] = val_loss

            components = phase.get('components')
            if isinstance(components, dict):
                normalized_components: Dict[str, List[float]] = {}
                for comp_name, comp_values in components.items():
                    normalized_components[str(comp_name)] = self._coerce_float_list(comp_values)
                phase['components'] = normalized_components

            normalized_history.append(phase)

        return normalized_history

    def get_embedding_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get the embedding snapshots captured during training.

        Returns:
            List of snapshot dicts, each with keys:
              - epoch: epoch number
              - phase: training phase name (e.g. 'warm-up', 'weighted')
              - embeddings: np.ndarray of shape (n_cells, z_dim)
        """
        return self._embedding_snapshots

    def set_effective_params(
        self,
        resolved_params: Optional[Dict[str, Any]] = None,
        *,
        include_declared_defaults: bool = True,
    ) -> Dict[str, Any]:
        """
        Persist the effective runtime parameter set used by this algorithm.

        Args:
            resolved_params: Optional resolved values (e.g. dynamic defaults,
                clamped values, fallback modes).
            include_declared_defaults: Whether to seed the map with defaults from
                get_hyperparameters() before applying overrides.

        Returns:
            The effective params dictionary stored on this instance.
        """
        effective: Dict[str, Any] = {}

        if include_declared_defaults:
            try:
                for hp in self.get_hyperparameters() or []:
                    name = getattr(hp, "name", None)
                    if name:
                        effective[name] = getattr(hp, "default", None)
            except Exception:
                # Best effort only: never block training.
                pass

        if isinstance(self.params, dict):
            effective.update(self.params)

        if resolved_params:
            effective.update(resolved_params)

        self._effective_params = effective
        return dict(self._effective_params)

    def get_effective_params(self) -> Dict[str, Any]:
        """
        Return effective runtime params if available, else raw overrides.
        """
        if isinstance(self._effective_params, dict) and self._effective_params:
            return dict(self._effective_params)
        return dict(self.params or {})

    @property
    def is_fitted(self) -> bool:
        """Check if the algorithm has been fitted."""
        return self._fitted

    def get_device(self) -> str:
        """
        Get the compute device to use (cuda/mps/cpu).
        
        Priority:
        1. Check params['device'] if explicitly set
        2. Check streamlit session_state if available
        3. Fall back to auto-detection (cuda > mps > cpu)
        
        Returns:
            'cuda', 'mps', or 'cpu'
        """
        import torch
        
        # Helper to detect best auto device
        def get_auto_device():
            """Réalise l'opération `get auto device` du module `algorithm_registry`.
            
            
            Args:
                Aucun argument explicite en dehors du contexte objet.
            
            Returns:
                `None` ou une valeur interne selon le flux d'exécution.
            """
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        
        # 1. Check if device is explicitly set in params
        device_param = self.params.get('device', None)
        if device_param is not None:
            if device_param == 'auto':
                return get_auto_device()
            elif device_param == 'cuda':
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    logger.warning("CUDA GPU requested but not available. Falling back to CPU.")
                    return 'cpu'
            elif device_param == 'mps':
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    logger.warning("MPS requested but not available. Falling back to CPU.")
                    return 'cpu'
            else:
                return 'cpu'
        
        # 2. Check streamlit session_state
        try:
            import streamlit as st
            effective_device = st.session_state.get('effective_device', None)
            if effective_device is not None:
                if effective_device == 'cuda' and not torch.cuda.is_available():
                    return 'cpu'
                if effective_device == 'mps':
                    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        return 'cpu'
                return effective_device
        except Exception:
            pass  # Streamlit not available or not in session
        
        # 3. Fall back to auto-detection
        return get_auto_device()

    def _set_seed(self, seed: int) -> None:
        """
        Set all random seeds for reproducibility.

        This is the centralized method for setting seeds in all algorithms.
        Call this at the beginning of fit() to ensure reproducible results.

        Args:
            seed: The random seed to use
        """
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class AlgorithmRegistry:
    """
    Registry for managing available algorithms.
    Singleton pattern to ensure single registry instance.
    """

    _instance = None
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}

    def __new__(cls):
        """Helper interne: new.
        
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            `None` ou une valeur interne selon le flux d'exécution.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, algorithm_class: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
        """
        Register a new algorithm.
        Can be used as a decorator.

        Args:
            algorithm_class: The algorithm class to register

        Returns:
            The registered class (for decorator usage)

        Example:
            @AlgorithmRegistry.register
            class MyAlgorithm(BaseAlgorithm):
                ...
        """
        info = algorithm_class.get_info()
        cls._algorithms[info.name] = algorithm_class
        logger.info(f"Registered algorithm: {info.name} ({info.display_name})")
        return algorithm_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseAlgorithm]]:
        """
        Get an algorithm by name.

        Args:
            name: The algorithm name

        Returns:
            The algorithm class or None if not found
        """
        return cls._algorithms.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseAlgorithm]]:
        """
        Get all registered algorithms.

        Returns:
            Dictionary mapping names to algorithm classes
        """
        return cls._algorithms.copy()

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, Type[BaseAlgorithm]]:
        """
        Get algorithms by category.

        Args:
            category: The category to filter by

        Returns:
            Dictionary of algorithms in the specified category
        """
        return {
            name: algo for name, algo in cls._algorithms.items()
            if algo.get_info().category == category
        }

    @classmethod
    def get_categories(cls) -> List[str]:
        """
        Get all unique categories.

        Returns:
            List of category names
        """
        return list(set(
            algo.get_info().category for algo in cls._algorithms.values()
        ))

    @classmethod
    def list_algorithms(cls) -> List[AlgorithmInfo]:
        """
        List all registered algorithms with their info.

        Returns:
            List of AlgorithmInfo objects
        """
        return [algo.get_info() for algo in cls._algorithms.values()]

    @classmethod
    def create(cls, name: str, params: Dict[str, Any] = None) -> BaseAlgorithm:
        """
        Create an instance of an algorithm.

        Args:
            name: The algorithm name
            params: Hyperparameters for the algorithm

        Returns:
            An instance of the algorithm

        Raises:
            ValueError: If algorithm not found
        """
        algorithm_class = cls.get(name)
        if algorithm_class is None:
            raise ValueError(f"Algorithm '{name}' not found. "
                           f"Available: {list(cls._algorithms.keys())}")
        return algorithm_class(params)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister an algorithm.

        Args:
            name: The algorithm name

        Returns:
            True if removed, False if not found
        """
        if name in cls._algorithms:
            del cls._algorithms[name]
            logger.info(f"Unregistered algorithm: {name}")
            return True
        return False

    @classmethod
    def clear(cls):
        """Clear all registered algorithms."""
        cls._algorithms.clear()
        logger.info("Cleared all registered algorithms")


def discover_algorithms(package_path: str = None):
    """
    Automatically discover and register algorithms from a package.

    Args:
        package_path: Path to search for algorithm modules
    """
    import importlib
    import pkgutil
    import sys

    if package_path is None:
        # Default to algorithms package
        from .. import algorithms
        package_path = algorithms.__path__
        package_name = algorithms.__name__
    else:
        package_name = package_path

    # Iterate through all modules in the package
    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        full_name = f"{package_name}.{modname}"
        try:
            importlib.import_module(full_name)
            logger.info(f"Loaded algorithm module: {full_name}")
        except Exception as e:
            logger.warning(f"Failed to load algorithm module {full_name}: {e}")
