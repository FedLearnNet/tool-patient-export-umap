from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.run_runfig import AppConfig, AppOutputConfig

from umap_pipeline import PipelineSettings


@dataclass
class UmapExportConfig(AppConfig):
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    aggregation: str = "last"
    min_feature_coverage: float = 0.5
    include_categorical: bool = True
    max_categories: int = 20
    scale: bool = True
    color_by: Optional[str] = ""
    exclude_color_feature: bool = True

    def to_settings(self) -> PipelineSettings:
        return PipelineSettings(
            n_neighbors=int(self.n_neighbors),
            min_dist=float(self.min_dist),
            metric=self.metric,
            random_state=int(self.random_state),
            aggregation=self.aggregation,
            min_feature_coverage=float(self.min_feature_coverage),
            include_categorical=bool(self.include_categorical),
            max_categories=int(self.max_categories),
            scale=bool(self.scale),
            color_by=self.color_by or "",
            exclude_color_feature=bool(self.exclude_color_feature),
        )


@dataclass
class UmapExportOutputConfig(AppOutputConfig):
    plot: Optional[Path] = None
    embedding: Optional[Path] = None
    feature_report: Optional[Path] = None
