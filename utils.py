import re


def clean_markdown(text: str) -> str:
    """
    Removing markdowns

    Parameters:
        text (str) -- input text

    Output:
        text (str) -- preprocessed text
    """
    # Remove Markdown link and image syntax
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove images
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove links

    # Remove inline code
    text = re.sub(r"`.*?`", "", text)

    # Remove bold and italics
    text = re.sub(r"\*\*.*?\*\*", "", text)  # Remove bold **
    text = re.sub(r"\*.*?\*", "", text)  # Remove italics *
    text = re.sub(r"__.*?__", "", text)  # Remove bold __
    text = re.sub(r"_.*?_", "", text)  # Remove italics _

    # Remove strikethrough
    text = re.sub(r"~~.*?~~", "", text)

    # Remove headers
    text = re.sub(r"\#{1,6}\s*", "", text)

    # Remove blockquotes
    text = re.sub(r"\>\s*", "", text)

    # Remove unordered list markers
    text = re.sub(r"[\*\-\+]\s+", "", text)

    # Remove ordered list markers
    text = re.sub(r"\d+\.\s+", "", text)

    # Remove horizontal rules
    text = re.sub(r"\-{3,}", "", text)

    # Remove remaining special characters used in markdown
    text = re.sub(r"\\.", "", text)  # Remove escaped characters like \*

    # Remove extra whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text
