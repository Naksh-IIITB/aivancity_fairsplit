from pathlib import Path

from pypdf import PdfReader


SOURCES = [
    (
        "/Users/naksh/Downloads/AIC_IIITB_outline_start_session1.pdf",
        "AIC_IIITB_outline_start_session1.txt",
    ),
    (
        "/Users/naksh/Downloads/AIC_IIITB_part2.pdf",
        "AIC_IIITB_part2.txt",
    ),
]


def main() -> None:
    out_dir = Path("assignment_extract")
    out_dir.mkdir(exist_ok=True)

    for source, name in SOURCES:
        reader = PdfReader(source)
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(f"\n--- PAGE {index} ---\n{page.extract_text() or ''}")
        (out_dir / name).write_text("\n".join(pages), encoding="utf-8")
        print(f"{name}: {len(reader.pages)} pages")


if __name__ == "__main__":
    main()
