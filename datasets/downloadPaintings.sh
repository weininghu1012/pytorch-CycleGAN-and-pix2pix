#!/bin/bash
# Basic for loop
names=['张大千荷花', '荷花', '齐白石荷花']
for name in $names
do
	googleimagesdownload --keywords $name --limit 100 --f jpg
done
echo All done
