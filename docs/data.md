# Dataset Schema

## Structure of JSON Data
- Parsed json contains a list of 5 dictionaries for each Instagram account.
- General Structure for Each Account:
    - `username`
    - `posts`
    - `id`

## Structure of DataFrame
- Tidy data containing information about each instagram post. Does not contain information about Google Vision annotations.
- Structure of DataFrame:
	- `likes`
	- `username`
	- `id`
	- `date`
	- `instagram_id`
	- `thumbnail_src`
	- `display_src`
	- `video`
	- `height`
	- `width`
	- `caption`