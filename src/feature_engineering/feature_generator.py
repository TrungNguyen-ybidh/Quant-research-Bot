"""
Feature Generator
Main orchestrator for feature engineering pipeline
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from src.utils.logger import info, success, error, warning
from src.features import (
    intrinsic_time,
    volatility,
    microstructure,
    trend,
    fx_factors,
    cross_asset,
    session,
    liquidity,
    indicators
)


class FeatureGenerator:
    """
    Feature Generator - Orchestrates feature computation from registry.
    
    Loads feature definitions from registry.yaml and computes features
    based on enabled flags and parameters.
    """
    
    def __init__(self, registry_path: str = "src/features/registry.yaml"):
        """
        Initialize Feature Generator.
        
        Args:
            registry_path: Path to feature registry YAML file
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.module_map = self._build_module_map()
        self.schema_version = self.registry.get('schema_version', '1.0')
        self.feature_version = self.registry.get('feature_version', '1.0')
        self.computed_features = {}  # Cache for computed features
        self.computation_metadata = {}  # Track computation metadata
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load feature registry from YAML file."""
        try:
            with open(self.registry_path, 'r') as f:
                registry = yaml.safe_load(f)
            info(f"Loaded feature registry from {self.registry_path}")
            return registry
        except FileNotFoundError:
            error(f"Registry file not found: {self.registry_path}")
            return {}
        except Exception as e:
            error(f"Error loading registry: {e}")
            return {}
    
    def _build_module_map(self) -> Dict[str, Any]:
        """Build mapping from module names to module objects."""
        return {
            'intrinsic_time': intrinsic_time,
            'volatility': volatility,
            'microstructure': microstructure,
            'trend': trend,
            'fx_factors': fx_factors,
            'cross_asset': cross_asset,
            'session': session,
            'liquidity': liquidity,
            'indicators': indicators
        }
    
    def _get_feature_function(self, module_name: str, function_name: str) -> Optional[Callable]:
        """
        Get feature function from module.
        
        Args:
            module_name: Name of the module
            function_name: Name of the function
        
        Returns:
            Function object or None if not found
        """
        if module_name not in self.module_map:
            error(f"Module '{module_name}' not found in module map")
            return None
        
        module = self.module_map[module_name]
        
        if not hasattr(module, function_name):
            error(f"Function '{function_name}' not found in module '{module_name}'")
            return None
        
        return getattr(module, function_name)
    
    def compute_feature(self, 
                       feature_name: str, 
                       df: pd.DataFrame,
                       category: Optional[str] = None,
                       use_cache: bool = True) -> Optional[pd.Series]:
        """
        Compute a single feature.
        
        Args:
            feature_name: Name of the feature (as defined in registry)
            df: Input DataFrame
            category: Optional category name (if None, searches all categories)
            use_cache: Whether to use cached result if available (default: True)
        
        Returns:
            Series with feature values or None if failed
        """
        # Check cache first
        if use_cache and feature_name in self.computed_features:
            return self.computed_features[feature_name]
        
        # Find feature in registry
        feature_def = None
        
        if category:
            if category in self.registry and feature_name in self.registry[category]:
                feature_def = self.registry[category][feature_name]
        else:
            # Search all categories
            for cat in self.registry:
                if feature_name in self.registry[cat]:
                    feature_def = self.registry[cat][feature_name]
                    break
        
        if not feature_def:
            warning(f"Feature '{feature_name}' not found in registry")
            return None
        
        # Check if enabled
        if not feature_def.get('enabled', True):
            info(f"Feature '{feature_name}' is disabled, skipping")
            return None
        
        # Get module and function
        module_name = feature_def.get('module')
        function_name = feature_def.get('function')
        
        if not module_name or not function_name:
            error(f"Feature '{feature_name}' missing module or function in registry")
            return None
        
        # Get function
        func = self._get_feature_function(module_name, function_name)
        if func is None:
            return None
        
        # Check dependencies
        dependencies = feature_def.get('depends_on', [])
        if dependencies:
            # Ensure dependencies are computed first
            for dep_name in dependencies:
                if dep_name not in self.computed_features:
                    # Try to compute dependency
                    dep_series = self.compute_feature(dep_name, df, use_cache=False)
                    if dep_series is not None:
                        self.computed_features[dep_name] = dep_series
                    else:
                        warning(f"Dependency '{dep_name}' for '{feature_name}' could not be computed")
        
        # Get parameters
        params = feature_def.get('params', {})
        
        # Compute feature
        try:
            result = func(df, **params)
            
            if isinstance(result, pd.Series):
                # Ensure index matches
                if len(result) != len(df):
                    warning(f"Feature '{feature_name}' returned series with different length")
                    result = result.reindex(df.index, method='ffill').fillna(0.0)
                
                # Cache result
                if use_cache:
                    self.computed_features[feature_name] = result
                
                return result
            else:
                error(f"Feature '{feature_name}' did not return a Series")
                return None
                
        except Exception as e:
            error(f"Error computing feature '{feature_name}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_dependency_graph(self, categories: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Build dependency graph for features.
        
        Args:
            categories: Optional list of categories to process
        
        Returns:
            Dictionary mapping feature names to their dependencies
        """
        dependency_graph = {}
        
        if categories is None:
            categories = [k for k in self.registry.keys() if k not in ['schema_version', 'feature_version']]
        
        for category in categories:
            if category not in self.registry:
                continue
            
            for feature_name, feature_def in self.registry[category].items():
                if not feature_def.get('enabled', True):
                    continue
                
                dependencies = feature_def.get('depends_on', [])
                if dependencies:
                    dependency_graph[feature_name] = dependencies
        
        return dependency_graph
    
    def _topological_sort(self, dependency_graph: Dict[str, List[str]], all_features: List[str]) -> List[str]:
        """
        Topologically sort features based on dependencies.
        
        Args:
            dependency_graph: Dictionary mapping features to dependencies
            all_features: List of all feature names to compute
        
        Returns:
            Sorted list of feature names (dependencies first)
        """
        # Build reverse graph (what depends on each feature)
        reverse_graph = {feat: [] for feat in all_features}
        for feat, deps in dependency_graph.items():
            if feat in all_features:
                for dep in deps:
                    if dep in all_features:
                        reverse_graph[dep].append(feat)
        
        # Kahn's algorithm for topological sort
        in_degree = {feat: len(dependency_graph.get(feat, [])) for feat in all_features}
        queue = [feat for feat in all_features if in_degree[feat] == 0]
        sorted_features = []
        
        while queue:
            feat = queue.pop(0)
            sorted_features.append(feat)
            
            for dependent in reverse_graph[feat]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Add any remaining features (circular dependencies or missing deps)
        remaining = [feat for feat in all_features if feat not in sorted_features]
        sorted_features.extend(remaining)
        
        return sorted_features
    
    def _get_all_enabled_features(self, categories: Optional[List[str]] = None) -> List[tuple]:
        """
        Get all enabled features with their categories.
        
        Args:
            categories: Optional list of categories to process
        
        Returns:
            List of (feature_name, category) tuples
        """
        features = []
        
        if categories is None:
            categories = [k for k in self.registry.keys() if k not in ['schema_version', 'feature_version']]
        
        for category in categories:
            if category not in self.registry:
                continue
            
            for feature_name, feature_def in self.registry[category].items():
                if feature_def.get('enabled', True):
                    features.append((feature_name, category))
        
        return features
    
    def compute_all_features(self, 
                            df: pd.DataFrame,
                            categories: Optional[List[str]] = None,
                            track_metadata: bool = True) -> pd.DataFrame:
        """
        Compute all enabled features from registry with dependency resolution.
        
        Args:
            df: Input DataFrame
            categories: Optional list of categories to compute (None = all)
            track_metadata: Whether to track computation metadata (default: True)
        
        Returns:
            DataFrame with all computed features
        """
        info("Starting feature computation with dependency resolution")
        
        # Reset caches
        self.computed_features = {}
        if track_metadata:
            self.computation_metadata = {
                'schema_version': self.schema_version,
                'feature_version': self.feature_version,
                'timestamp': datetime.now().isoformat(),
                'computed_features': {},
                'parameters': {}
            }
        
        # Get all enabled features
        all_features = self._get_all_enabled_features(categories)
        feature_names = [feat[0] for feat in all_features]
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(categories)
        
        # Topologically sort features
        sorted_features = self._topological_sort(dependency_graph, feature_names)
        
        info(f"Computation order determined: {len(sorted_features)} features")
        
        # Create feature DataFrame
        features_df = pd.DataFrame(index=df.index)
        computed_count = 0
        failed_count = 0
        
        # Compute features in dependency order
        for feature_name in sorted_features:
            # Find category
            category = None
            for feat_name, cat in all_features:
                if feat_name == feature_name:
                    category = cat
                    break
            
            if category is None:
                continue
            
            # Check if already computed (from cache)
            if feature_name in self.computed_features:
                features_df[feature_name] = self.computed_features[feature_name]
                computed_count += 1
                continue
            
            # Compute feature
            start_time = datetime.now()
            feature_series = self.compute_feature(feature_name, df, category)
            end_time = datetime.now()
            
            if feature_series is not None:
                features_df[feature_name] = feature_series
                self.computed_features[feature_name] = feature_series
                computed_count += 1
                
                # Track metadata
                if track_metadata:
                    # Convert numpy types to Python native types for JSON serialization
                    null_count = feature_series.isna().sum()
                    if hasattr(null_count, 'item'):  # numpy scalar
                        null_count = int(null_count.item())
                    else:
                        null_count = int(null_count)
                    
                    shape = feature_series.shape
                    if isinstance(shape, tuple):
                        shape = tuple(int(x) for x in shape)
                    
                    self.computation_metadata['computed_features'][feature_name] = {
                        'category': category,
                        'computation_time': float((end_time - start_time).total_seconds()),
                        'shape': shape,
                        'null_count': null_count
                    }
            else:
                failed_count += 1
        
        success(f"Feature computation complete: {computed_count} computed, {failed_count} failed")
        
        # Add metadata as DataFrame attribute
        if track_metadata:
            features_df.attrs['metadata'] = self.computation_metadata
        
        return features_df
    
    def compute_category_features(self, 
                                 df: pd.DataFrame,
                                 category: str) -> pd.DataFrame:
        """
        Compute all features from a specific category.
        
        Args:
            df: Input DataFrame
            category: Category name
        
        Returns:
            DataFrame with features from the category
        """
        return self.compute_all_features(df, categories=[category])
    
    def list_features(self, category: Optional[str] = None) -> List[str]:
        """
        List all features in registry.
        
        Args:
            category: Optional category name (None = all categories)
        
        Returns:
            List of feature names
        """
        features = []
        
        if category:
            if category in self.registry:
                features = list(self.registry[category].keys())
        else:
            for cat in self.registry:
                features.extend(self.registry[cat].keys())
        
        return features
    
    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a feature.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            Dictionary with feature information or None if not found
        """
        for category in self.registry:
            if feature_name in self.registry[category]:
                return self.registry[category][feature_name]
        return None
    
    def enable_feature(self, feature_name: str, category: Optional[str] = None):
        """
        Enable a feature in the registry.
        
        Args:
            feature_name: Name of the feature
            category: Optional category name (if None, searches all)
        """
        feature_def = None
        
        if category:
            if category in self.registry and feature_name in self.registry[category]:
                feature_def = self.registry[category][feature_name]
        else:
            for cat in self.registry:
                if feature_name in self.registry[cat]:
                    feature_def = self.registry[cat][feature_name]
                    category = cat
                    break
        
        if feature_def:
            feature_def['enabled'] = True
            info(f"Enabled feature '{feature_name}'")
        else:
            warning(f"Feature '{feature_name}' not found in registry")
    
    def disable_feature(self, feature_name: str, category: Optional[str] = None):
        """
        Disable a feature in the registry.
        
        Args:
            feature_name: Name of the feature
            category: Optional category name (if None, searches all)
        """
        feature_def = None
        
        if category:
            if category in self.registry and feature_name in self.registry[category]:
                feature_def = self.registry[category][feature_name]
        else:
            for cat in self.registry:
                if feature_name in self.registry[cat]:
                    feature_def = self.registry[cat][feature_name]
                    category = cat
                    break
        
        if feature_def:
            feature_def['enabled'] = False
            info(f"Disabled feature '{feature_name}'")
        else:
            warning(f"Feature '{feature_name}' not found in registry")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get computation metadata.
        
        Returns:
            Dictionary with computation metadata
        """
        return self.computation_metadata
    
    def get_schema_version(self) -> str:
        """Get schema version from registry."""
        return self.schema_version
    
    def get_feature_version(self) -> str:
        """Get feature version from registry."""
        return self.feature_version


def generate_features(df: pd.DataFrame,
                     registry_path: str = "src/features/registry.yaml",
                     categories: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to generate features.
    
    Args:
        df: Input DataFrame
        registry_path: Path to feature registry YAML file
        categories: Optional list of categories to compute
    
    Returns:
        DataFrame with computed features
    """
    generator = FeatureGenerator(registry_path=registry_path)
    return generator.compute_all_features(df, categories=categories)

