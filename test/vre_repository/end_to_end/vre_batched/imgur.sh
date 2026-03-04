#!/usr/bin/env bash

# Upload images to imgur and output their URLs to stdout.
# Uses imgur's browser upload endpoint (the v3 API stopped working in 2025).

# Function to output usage instructions
function usage {
	echo "Usage: $(basename $0) [<filename> [...]]" >&2
	echo
	echo "Upload images to imgur and output their new URLs to stdout." >&2
	echo
	echo "A filename can be - to read from stdin. If no filename is given, stdin is read." >&2
	echo
	echo "If xsel, xclip, pbcopy, or clip is available," >&2
	echo "the URLs are put on the X selection or clipboard for easy pasting." >&2
}

# Function to upload a file via imgur's browser upload endpoint
function upload {
	curl -s -X POST "https://imgur.com/upload" \
		-H "Referer: https://imgur.com/upload" \
		-F "Filedata=$1"
}

# Check arguments
if [ "$1" == "-h" -o "$1" == "--help" ]; then
	usage
	exit 0
elif [ $# -eq 0 ]; then
	echo "No file specified; reading from stdin" >&2
	exec "$0" -
fi

# Check curl is available
type curl &>/dev/null || {
	echo "Couldn't find curl, which is required." >&2
	exit 17
}

clip=""
errors=false

# Loop through arguments
while [ $# -gt 0 ]; do
	file="$1"
	shift

	# Check file exists
	if [ "$file" != "-" -a ! -f "$file" ]; then
		echo "File '$file' doesn't exist; skipping" >&2
		errors=true
		continue
	fi

	# Upload the image
	response=$(upload "@$file")

	if [ $? -ne 0 ] || [ -z "$response" ]; then
		echo "Upload failed" >&2
		errors=true
		continue
	fi

	# Parse JSON response to extract the hash
	hash=$(echo "$response" | grep -o '"hash":"[^"]*"' | head -1 | cut -d'"' -f4)

	if [ -z "$hash" ]; then
		echo "Error from imgur:" >&2
		echo "$response" >&2
		errors=true
		continue
	fi

	url="https://i.imgur.com/${hash}.png"
	echo "$url"
	delete_hash=$(echo "$response" | grep -o '"deletehash":"[^"]*"' | head -1 | cut -d'"' -f4)
	echo "Delete page: https://imgur.com/delete/$delete_hash" >&2

	# Append the URL to a string so we can put them all on the clipboard later
	clip+="$url"
	if [ $# -gt 0 ]; then
		clip+=$'\n'
	fi
done

# Put the URLs on the clipboard if we can
if type pbcopy &>/dev/null; then
	echo -n "$clip" | pbcopy
elif type clip &>/dev/null; then
	echo -n "$clip" | clip
elif [ $DISPLAY ]; then
	if type xsel &>/dev/null; then
		echo -n "$clip" | xsel -i
	elif type xclip &>/dev/null; then
		echo -n "$clip" | xclip
	else
		echo "Haven't copied to the clipboard: no xsel or xclip" >&2
	fi
else
	echo "Haven't copied to the clipboard: no \$DISPLAY or pbcopy or clip" >&2
fi

if $errors; then
	exit 1
fi
