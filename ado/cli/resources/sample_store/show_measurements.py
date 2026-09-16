# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing

from rich.status import Status

from ado.cli.models.parameters import AdoShowMeasurementsCommandParameters
from ado.cli.models.types import (
    AdoShowMeasurementsSupportedPropertyFormats,
)
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.dataframes import df_to_output
from ado.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    INFO,
    console_print,
    magenta,
)
from ado.core.samplestore.base import SampleStore
from ado.metastore.base import ResourceDoesNotExistError

if typing.TYPE_CHECKING:
    from ado.schema.entity import Entity


def show_sample_store_measurements(
    parameters: AdoShowMeasurementsCommandParameters,
) -> None:
    """Show measurements for all entities in a samplestore.

    Args:
        parameters: The command parameters including the samplestore identifier,
            output format, property format, and optional filters.
    """
    import pandas as pd

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            sample_store = SampleStore.from_identifier(
                identifier=parameters.resource_id,
                metastore=sql_store,  # type: ignore[arg-type]
            )
        except ResourceDoesNotExistError:
            status.stop()
            raise

        status.update("Fetching measurements")
        entities: list[Entity] = sample_store.get_entities(require_measurements=True)

    measured_entities = [e for e in entities if len(e.observedPropertyValues) > 0]

    if not measured_entities:
        console_print(
            f"{INFO}Nothing was returned for "
            f"[i]property format {magenta(parameters.measurements_property_format.value)}[/i] "
            f"in [i]samplestore {magenta(parameters.resource_id)}[/i].",
            stderr=True,
        )
        return

    references = list(
        {ref for e in measured_entities for ref in e.experimentReferences}
    )

    if (
        parameters.measurements_property_format
        == AdoShowMeasurementsSupportedPropertyFormats.OBSERVED
    ):
        output_df = pd.DataFrame(
            data=[
                e.seriesRepresentation(experimentReferences=references)
                for e in measured_entities
            ]
        )
    else:
        data = []
        for e in measured_entities:
            data.extend(e.experimentSeries(experimentReferences=references))
        output_df = pd.DataFrame(data)

    if parameters.properties:
        df_column_set = set(output_df.columns)
        properties_set = set(parameters.properties)
        if properties_set.issubset(df_column_set):
            parameters.properties.insert(0, "identifier")
            output_df = output_df[parameters.properties]

    df_to_output(
        df=output_df,
        output_format=parameters.measurements_output_format.value,
        output_file=parameters.output_file,
        no_trunc=parameters.no_trunc,
    )
