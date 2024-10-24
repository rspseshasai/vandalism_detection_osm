from datetime import datetime, timezone
import re
import osmium
import pandas as pd
from logger.logger_config import logger


# Custom handler to read and store changesets with relevant comments
class MyOSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.changeset_data = []
        self.count = 0
        self.file_count = 0  # To keep track of the number of files saved

        # Define the time period: Last year (Aug 2023 to Aug 2024) with timezone awareness (UTC)
        self.start_date = datetime(2023, 8, 1, tzinfo=timezone.utc)
        self.end_date = datetime(2024, 8, 31, tzinfo=timezone.utc)

    def changeset(self, cs):
        # Only process changesets within the specified time period
        if self.start_date <= cs.closed_at <= self.end_date:
            comment = None
            original_changeset_id = None

            # Check for tags and particularly for comments indicating vandalism or reverts
            if cs.tags:
                for tag in cs.tags:
                    if tag.k == 'comment':
                        comment = tag.v
                        # Filter only comments containing the words "vandalism" or "revert"
                        if re.search(r'vandalism|revert', comment, re.IGNORECASE):
                            # Try to extract a changeset ID from the comment (for reverted changeset)
                            match = re.search(r'\b\d{6,}\b', comment)
                            if match:
                                original_changeset_id = match.group(0)  # Extracted changeset ID
                            # Create a dictionary for filtered changeset data
                            changeset_info = {
                                "changeset_id": cs.id,
                                "comment": comment,
                                "original_changeset_id": original_changeset_id,
                                "label": "vandalism"  # Always label as vandalism
                            }
                            self.changeset_data.append(changeset_info)
                            self.count += 1

            # Log progress every 100,000 changesets
            if self.count % 100000 == 0:
                logger.info(f"Processed {self.count} changesets so far.")
                # Save the current batch to a Parquet file
                self.save_to_parquet()

    def save_to_parquet(self):
        if not self.changeset_data:
            return  # Avoid saving if there's no data to save

        # Convert current changeset data to a DataFrame with only the needed columns
        changeset_df = pd.DataFrame(self.changeset_data, columns=["changeset_id", "comment", "original_changeset_id", "label"])

        # Define the filename for the Parquet file
        parquet_file_path = f"../data/changeset_data/osm_filtered_changesets_{self.file_count}.parquet"

        # Save to Parquet file
        changeset_df.to_parquet(parquet_file_path, index=False)

        logger.info(f"Successfully saved {len(changeset_df)} filtered changesets to {parquet_file_path}.")

        # Reset the changeset_data and increment file_count
        self.changeset_data = []
        self.file_count += 1


# Path to your OSM file
osm_file = "E:\\HeidelbergUniversity\\Thesis\\HeiGit\\data\\changesets-240930\\changesets-240930.osm"

logger.info("Start processing...")
handler = MyOSMHandler()

# Apply the handler to the OSM file
try:
    handler.apply_file(osm_file)
except StopIteration:
    logger.info("Done processing the OSM file")

# Save any remaining filtered changeset data that wasn't saved yet
if handler.changeset_data:
    handler.save_to_parquet()

logger.info(f"Successfully processed a total of {handler.count} filtered changesets.")
