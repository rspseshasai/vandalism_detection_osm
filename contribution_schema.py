import pyarrow as pa


def get_osm_contribution_schema():
    return pa.schema([
        ('user_id', pa.int32()),
        ('valid_from', pa.timestamp('ms')),
        ('valid_to', pa.timestamp('ms')),
        ('osm_type', pa.string()),
        ('osm_id', pa.string()),
        ('osm_version', pa.int32()),
        ('contrib_type', pa.string()),
        ('members', pa.list_(pa.struct([
            ('type', pa.string()),
            ('id', pa.int64()),
            ('role', pa.string()),
            ('geometry', pa.binary())
        ]))),
        ('status', pa.string()),
        ('changeset', pa.struct([
            ('id', pa.int64()),
            ('timestamp', pa.timestamp('ms')),
            ('tags', pa.map_(pa.string(), pa.string())),
            ('hashtags', pa.list_(pa.string())),
            ('editor', pa.string())
        ])),
        ('tags', pa.map_(pa.string(), pa.string())),
        ('tags_before', pa.map_(pa.string(), pa.string())),
        ('map_features', pa.struct([
            ('aerialway', pa.bool_()),
            ('aeroway', pa.bool_()),
            ('amenity', pa.bool_()),
            ('barrier', pa.bool_()),
            ('boundary', pa.bool_()),
            ('building', pa.bool_()),
            ('craft', pa.bool_()),
            ('emergency', pa.bool_()),
            ('geological', pa.bool_()),
            ('healthcare', pa.bool_()),
            ('highway', pa.bool_()),
            ('historic', pa.bool_()),
            ('landuse', pa.bool_()),
            ('leisure', pa.bool_()),
            ('man_made', pa.bool_()),
            ('military', pa.bool_()),
            # Add any other map features as needed
        ])),
        ('area', pa.int64()),
        ('area_delta', pa.int64()),
        ('length', pa.int64()),
        ('length_delta', pa.int64()),
        ('xzcode', pa.struct([
            ('level', pa.int32()),
            ('code', pa.int64())
        ])),
        ('country_iso_a3', pa.list_(pa.string())),
        ('bbox', pa.struct([
            ('xmin', pa.float64()),
            ('ymin', pa.float64()),
            ('xmax', pa.float64()),
            ('ymax', pa.float64())
        ])),
        ('xmin', pa.float64()),
        ('xmax', pa.float64()),
        ('ymin', pa.float64()),
        ('ymax', pa.float64()),
        ('centroid', pa.struct([
            ('x', pa.float64()),
            ('y', pa.float64())
        ])),
        ('quadkey_z10', pa.string()),
        ('h3_r5', pa.uint64()),
        ('geometry_type', pa.string()),
        ('geometry_valid', pa.bool_()),
        ('geometry', pa.string()),  # Assuming geometry is stored as WKT or similar string format
        ('vandalism', pa.bool_())
    ])
