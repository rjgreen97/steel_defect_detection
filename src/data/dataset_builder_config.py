from pydantic import Field

from src.config.base_config import BaseConfig


class DatasetBuilderConfig(BaseConfig):
    """XML parsing tags for the dataset builder."""

    top_left_x: str = Field(
        default="bndbox/xmin", description="XML tag for top left x coordinate value"
    )

    top_left_y: str = Field(
        default="bndbox/ymin", description="XML tag for top left y coordinate value"
    )

    bottom_right_x: str = Field(
        default="bndbox/xmax", description="XML tag for bottom right x coordinate value"
    )

    bottom_right_y: str = Field(
        default="bndbox/ymax", description="XML tag for bottom right y coordinate value"
    )

    filename_tag: str = Field(
        default="filename", description="XML tag for image filename"
    )

    image_tag: str = Field(default="images", description="XML tag for image")

    object_tag: str = Field(default="object", description="XML tag for object")

    annotations_dir: str = Field(
        default="annotations", description="Directory where annotations are stored"
    )

    scratches_annotation: str = Field(
        default="scratches",
        description="Annotation name for scratches",
    )
