In OpenStreetMap (OSM), **contributions** and **changesets** are related but distinct concepts:

### 1. **Contributions**:

- **Definition**: Contributions refer to the individual edits or modifications made by a user to OSM data. These edits
  can include creating, modifying, or deleting map features such as roads, buildings, points of interest (POIs), or
  geographic boundaries.
- **Granularity**: Each contribution typically represents a single action or update, like adding a new road or changing
  the tag of a building.
- **Data Types**: Contributions are applied to **nodes** (points), **ways** (lines or polygons), and **relations** (
  groupings of objects) within the OSM database. Each of these changes represents one contribution.

### 2. **Changesets**:

- **Definition**: A changeset is a collection of contributions that are grouped together and uploaded to the OSM
  database in one batch. A single changeset can contain multiple contributions made by a user over a certain period.
- **Time-bound**: Changesets are time-limited and have an opening and closing time. Users can accumulate edits within
  this period and submit them as one logical unit.
- **Metadata**: Changesets include metadata such as the user who made the edits, a comment describing the changes,
  geographic bounding boxes of where changes were made, and the date/time of the edits. They provide context to the
  edits being uploaded.

### Key Differences:

- **Scope**: A contribution is a single edit, whereas a changeset is a container for multiple edits (contributions).
- **Purpose**: Contributions represent granular changes to map data, while changesets bundle those changes together for
  easier review, tracking, and analysis.
- **Metadata**: Changesets come with additional metadata (e.g., comments, bounding boxes) that help reviewers understand
  the broader context of the contributions.

In summary, **contributions** are the actual edits made to the map, and **changesets** are a way of organizing and
submitting those contributions in batches.

In OpenStreetMap (OSM), contributions are grouped into a **changeset** based on several factors, which provide logical
or contextual grouping for edits. These factors ensure that related contributions are bundled together in one changeset.
Here's how contributions are typically related within a changeset:

### 1. **Session-Based Grouping**:

- Changesets are often created during a user’s editing session. While a user is editing the map, all the changes made
  during that session are grouped into a single changeset. Once the user finishes editing and uploads the changes, the
  changeset is closed.
- **Example**: A user might add several new roads and update some building names within the same area during a single
  session. These actions will be grouped under the same changeset.

### Conclusion:

Contributions within a changeset are typically related by **time**, **geography**, **purpose**, and **logical grouping
of edits**. Changesets provide a way to bundle related contributions together, making it easier to review, manage, and
track edits on the OSM platform.