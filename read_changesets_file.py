import osmium


# Custom handler to read and print changesets with tags
class MyOSMHandler(osmium.SimpleHandler):
    def __init__(self, max_count):
        super().__init__()
        self.count = 0
        self.max_count = max_count

    def changeset(self, cs):
        # Check if changeset has any tags
        if cs.tags and self.count < self.max_count:
            # Print changeset details
            print(f"\nChangeset ID: {cs.id}")
            print(f"Created at: {cs.created_at}")
            print(f"Closed at: {cs.closed_at}")
            print(f"User: {cs.user}")
            print(f"Number of changes: {cs.num_changes}")
            print(f"Bounds: {cs.bounds}")

            # Print tags if present
            print("Tags:")
            for tag in cs.tags:
                print(f"  {tag.k}: {tag.v}")

            self.count += 1
        elif self.count >= self.max_count:
            raise StopIteration  # Stop after reaching the max count


# Path to your OSM file
osm_file = "E:\HeidelbergUniversity\Thesis\HeiGit\data\changesets-240930\changesets-240930.osm"

# Number of changesets with tags you want to read (like head in pandas)
N = 10

# Initialize the handler and apply it to the OSM file
handler = MyOSMHandler(max_count=N)

try:
    handler.apply_file(osm_file)
except StopIteration:
    print(f"\nDisplayed the first {N} changesets with tags successfully.")
