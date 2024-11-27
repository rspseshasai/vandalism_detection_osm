import osmium
import pandas as pd
from logger_config import logger


# Custom handler to read and store changesets with tags
class MyOSMHandler(osmium.SimpleHandler):
    def __init__(self, min_id, max_id):
        super().__init__()
        self.changeset_data = []
        self.min_id = min_id
        self.max_id = max_id
        self.count = 0
        self.file_count = 0  # To keep track of the number of files saved

    def changeset(self, cs):
        # Only process changesets within the specified ID range
        if self.min_id <= cs.id <= self.max_id:
            # Attempt to extract bounds with error handling
            try:
                min_lat = cs.bounds.bottom_left.lat if cs.bounds else None
                min_lon = cs.bounds.bottom_left.lon if cs.bounds else None
                max_lat = cs.bounds.top_right.lat if cs.bounds else None
                max_lon = cs.bounds.top_right.lon if cs.bounds else None
            except osmium._osmium.InvalidLocationError:
                min_lat, min_lon, max_lat, max_lon = None, None, None, None

            # Initialize fields for tags
            comment = None
            created_by = None

            # Create a dictionary for the tags
            if cs.tags:
                for tag in cs.tags:
                    if tag.k == 'comment':
                        comment = tag.v
                    elif tag.k == 'created_by':
                        created_by = tag.v

            # Create a dictionary for the changeset data
            changeset_info = {
                "changeset_id": cs.id,
                "created_at": cs.created_at,
                "closed_at": cs.closed_at,
                "user": cs.user,
                "user_id": cs.uid,
                "num_changes": cs.num_changes,
                "min_lat": min_lat,
                "min_lon": min_lon,
                "max_lat": max_lat,
                "max_lon": max_lon,
                "comment": comment,
                "created_by": created_by
            }
            self.changeset_data.append(changeset_info)
            self.count += 1

            # Log progress every 1,000,000 changesets
            if self.count % 1000000 == 0:
            # if self.count % 100 == 0:
                logger.info(f"Processed until changeset ID: {cs.id}")

                # Save the current batch to a Parquet file
                self.save_to_parquet()

    def save_to_parquet(self):
        # Convert current changeset data to a DataFrame
        changeset_df = pd.DataFrame(self.changeset_data)

        # Define the filename for the Parquet file
        parquet_file_path = f"../../data/changeset_data/last_2_years/osm_filtered_changesets_{self.file_count}.parquet"

        # Save to Parquet file
        changeset_df.to_parquet(parquet_file_path, index=False)

        logger.info(f"Successfully saved {len(changeset_df)} changesets to {parquet_file_path}.")

        # Reset the changeset_data and increment file_count
        self.changeset_data = []
        self.file_count += 1


# Path to your OSM file
osm_file = "D:\\HeidelbergUniversity\\Thesis\\HeiGit\\data\\changesets-240930\\changesets-240930.osm"

# Initialize the handler for the specified ID range
# min_changeset_id = 120000000 # 2022-04-21
# max_changeset_id = 158900000 # 2024-11-08

min_changeset_id = 120000000 # 2022-04-21
max_changeset_id = 158900000 # 2024-11-08

logger.info("Start processing...")
handler = MyOSMHandler(min_changeset_id, max_changeset_id)

# Apply the handler to the OSM file
try:
    handler.apply_file(osm_file)
except StopIteration:
    logger.info("Done processing the OSM file")

# Save any remaining changeset data that wasn't saved yet
if handler.changeset_data:
    handler.save_to_parquet()

logger.info(f"Successfully processed a total of {handler.count} changesets.")
