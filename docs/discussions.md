## 02-Nov-2024
### Discussion points with Benni
1. Showcase the results of both changeset and contribution models.
2. Tell that most of the features in contribution data are very similar. Especially all contributions under same changeset id. It might be the potential reason for high accuracy.
3. How to get recent labelled data? (for both changeset and contribution)
4. User details already extracted during the OSM data insights workshop? If not how to get the user features?
5. https://osmcha.org/api-docs/

### TODO:
- All contributions of a changeset should only be in one class (i.e., either in training data or in testing data)
- All contributions in a particular Geo-spatial region should only be in one class (i.e., either in training data or in testing data)
- CV based on changesets
- CV based on location
- Feature importance (and recursively remove the most important features until the model's performance won't decline anymore).
- GIScience/ChangesetMD - to convert .osm file to .parquet file.
- Use the trained model to predict on all unseen changesets & contributions data from 2023, 2024.
  - Graph between timeline (from 2023 to 2024) and vandalism: for both changeset and contributions data.
  - Heap map of world map with vandalism contributions (good to have).

### TODO Later:
- Outlier analysis.