## DuckDB Commands for Common Queries

### Count Rows Based on Comment Content

#### Example 1: Count Rows with 'revert' in Comment

```sql
SELECT COUNT(*)
FROM 'contri_test_0.parquet'
WHERE contains (changeset.tags['comment'][1], 'revert');
```

#### Example 2: Count Rows with 'fixed' in Comment

```sql
SELECT COUNT(*)
FROM 'contri_test_0.parquet'
WHERE contains (changeset.tags['comment'][1], 'fixed');
```

#### Example 3: Count Rows with 'revert' in Comment

```sql
SELECT COUNT(*)
FROM 'ohsome_contributions_2023_09.parquet'
WHERE contains (changeset_tags['comment'][1], 'revert');
```

### Retrieve Records by Specific Criteria

#### Example 1: Retrieve Records by Specific changeset ID

```sql
SELECT *
FROM 'ohsome_contributions_2023_09.parquet'
WHERE changeset_id='141719077';
```

#### Example 2: Retrieve Records by Specific changeset ID

```sql
SELECT *
FROM 'contri_test_0.parquet'
WHERE changeset.id='153074871';
```

#### Example 3: Retrieve Records by Another Specific changeset ID

```sql
SELECT *
FROM 'contri_test_0.parquet'
WHERE changeset.id='154329021';
```

#### Example 4: Retrieve Records by Specific User ID

```sql
SELECT *
FROM 'contri_test_0.parquet'
WHERE user_id='11374';
```

### Analyze Comments

#### Example 1: Count Non-Null Values of 'comment' Field

```sql
SELECT COUNT(changeset_tags['comment'][1])
FROM 'ohsome_contributions_2023_09.parquet';
```

#### Example 2: Retrieve Specific Comments Containing 'revert'

```sql
SELECT changeset_id, changeset_tags['comment'][1]
FROM 'ohsome_contributions_2023_09.parquet'
WHERE contains (changeset_tags['comment'][1], 'revert') LIMIT 1400;
```

### Describe Schema

#### Example 1: Describe Schema of Filtered Contributions File

```sql
DESCRIBE SELECT * FROM 'filtered_contributions_part_0.parquet';
```