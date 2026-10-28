import tempfile
from pathlib import Path

import pandas as pd
from pyfedappwrap.types.export.base_exporter import BaseExporterAPP

from config import UmapExportConfig, UmapExportOutputConfig
from plot import render_html
from umap_pipeline import run_pipeline


class PatientUmapExportAPP(BaseExporterAPP[UmapExportConfig, UmapExportOutputConfig]):
    """
    Patient UMAP Export (Export app)
    - Receives the patient data export of a cohort
    - Pivots it to one row per patient and encodes the features
    - Embeds the patients with UMAP and renders an interactive scatter plot
    """

    def export(self, df: pd.DataFrame) -> UmapExportOutputConfig:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The export app expects the patient export as a table")

        settings = self.config.to_settings()
        self.logger.info(f"Received export with {len(df)} rows and columns {list(df.columns)}")

        result = run_pipeline(df, settings, log=self.logger.info)
        for note in result.notes:
            self.logger.warning(note)

        out_dir = Path(tempfile.mkdtemp(prefix="patient-umap-"))
        plot_path = out_dir / "plot.html"
        plot_path.write_text(render_html(result, settings), encoding="utf-8")
        embedding_path = out_dir / "embedding.csv"
        result.embedding.to_csv(embedding_path, index=False)
        report_path = out_dir / "feature_report.csv"
        result.features_frame().to_csv(report_path, index=False)

        self.send_metric("patients", float(len(result.embedding)), 0.0)
        self.send_metric("features", float(len(result.used_features)), 0.0)
        self.logger.info(f"Embedded {len(result.embedding)} patients using {len(result.used_features)} features")

        return UmapExportOutputConfig(plot=plot_path, embedding=embedding_path, feature_report=report_path)
