class XesFields:
    """
    Common xes fields that may be present in a xes log.
    """
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"
    RESOURCE_COLUMN = "org:resource"


class DataFrameFields:
    """
    Common column names that may be present in a csv log.
    """
    CASE_COLUMN = "CaseID"
    ACTIVITY_COLUMN = "Activity"
    TIMESTAMP_COLUMN = "Timestamp"
    RESOURCE_COLUMN = "Resource"