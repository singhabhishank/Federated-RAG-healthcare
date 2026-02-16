#!/usr/bin/env python3
"""
Extract structured text fields from PubMed Central XML files.
Extracts: Title, Abstract, Sections (Introduction, Methods, Results, Discussion), and Metadata (journal, year)
"""

import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import re


def get_text(element) -> str:
    """Extract all text content from an XML element recursively."""
    if element is None:
        return ""
    
    text_parts = []
    
    # Get direct text
    if element.text:
        text_parts.append(element.text.strip())
    
    # Get text from all children
    for child in element:
        child_text = get_text(child)
        if child_text:
            text_parts.append(child_text)
        # Add tail text (text after the child element)
        if child.tail:
            text_parts.append(child.tail.strip())
    
    return " ".join(text_parts)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_title(root) -> str:
    """Extract article title."""
    # Try different possible locations for title
    title = None
    
    # Look in front/article-meta/title-group/article-title
    for path in [
        ".//front//article-title",
        ".//article-title",
        ".//title-group/article-title"
    ]:
        title_elem = root.find(path)
        if title_elem is not None:
            title = get_text(title_elem)
            if title:
                break
    
    return clean_text(title) if title else ""


def extract_abstract(root) -> str:
    """Extract abstract text."""
    abstract = None
    
    # Look in front/article-meta/abstract
    for path in [
        ".//front//abstract",
        ".//abstract"
    ]:
        abstract_elem = root.find(path)
        if abstract_elem is not None:
            abstract = get_text(abstract_elem)
            if abstract:
                break
    
    return clean_text(abstract) if abstract else ""


def extract_sections(root) -> Dict[str, str]:
    """Extract sections: Introduction, Methods, Results, Discussion."""
    sections = {
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": ""
    }
    
    # Find all section elements in the body
    body = root.find(".//body")
    if body is None:
        return sections
    
    # Find all sec elements
    sec_elements = body.findall(".//sec")
    
    for sec in sec_elements:
        # Get section title
        title_elem = sec.find("title")
        if title_elem is None:
            continue
        
        section_title = get_text(title_elem).lower()
        section_text = get_text(sec)
        
        # Match section titles to our target sections
        if any(keyword in section_title for keyword in ["introduction", "background"]):
            if not sections["Introduction"]:
                sections["Introduction"] = clean_text(section_text)
        elif any(keyword in section_title for keyword in ["method", "materials", "experimental"]):
            if not sections["Methods"]:
                sections["Methods"] = clean_text(section_text)
        elif any(keyword in section_title for keyword in ["result", "finding"]):
            if not sections["Results"]:
                sections["Results"] = clean_text(section_text)
        elif any(keyword in section_title for keyword in ["discussion", "conclusion"]):
            if not sections["Discussion"]:
                sections["Discussion"] = clean_text(section_text)
    
    return sections


def extract_metadata(root) -> Dict[str, str]:
    """Extract metadata: journal and year."""
    metadata = {
        "journal": "",
        "year": ""
    }
    
    # Extract journal name
    journal_paths = [
        ".//front//journal-title",
        ".//journal-title",
        ".//source"
    ]
    
    for path in journal_paths:
        journal_elem = root.find(path)
        if journal_elem is not None:
            journal = get_text(journal_elem)
            if journal:
                metadata["journal"] = clean_text(journal)
                break
    
    # Extract year
    year_paths = [
        ".//front//pub-date/year",
        ".//pub-date/year",
        ".//year"
    ]
    
    for path in year_paths:
        year_elem = root.find(path)
        if year_elem is not None:
            year = get_text(year_elem)
            if year:
                metadata["year"] = clean_text(year)
                break
    
    return metadata


def extract_pmc_id(root) -> str:
    """Extract PMC ID from the XML."""
    # Try to find PMC ID in various locations
    pmc_id = ""
    
    # Look for article-id with pub-id-type="pmc"
    for article_id in root.findall(".//article-id"):
        pub_id_type = article_id.get("pub-id-type", "")
        if pub_id_type == "pmc":
            pmc_id = get_text(article_id)
            break
    
    # If not found, try to extract from filename or other sources
    return clean_text(pmc_id)


def process_xml_file(xml_path: Path) -> Optional[Dict]:
    """Process a single XML file and extract all fields."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract PMC ID (try from filename if not in XML)
        pmc_id = extract_pmc_id(root)
        if not pmc_id:
            # Extract from filename (e.g., PMC1043859.xml -> PMC1043859)
            pmc_id = xml_path.stem
        
        # Extract all fields
        result = {
            "pmc_id": pmc_id,
            "title": extract_title(root),
            "abstract": extract_abstract(root),
            "sections": extract_sections(root),
            "metadata": extract_metadata(root)
        }
        
        return result
    
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return None


def process_all_xml_files(dataset_dir: str, output_file: str = "extracted_data.json"):
    """Process all XML files in the dataset directory."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_dir} does not exist")
        return
    
    # Find all XML files
    xml_files = list(dataset_path.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process each file
    all_results = []
    processed = 0
    failed = 0
    
    for xml_file in xml_files:
        result = process_xml_file(xml_file)
        if result:
            all_results.append(result)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} files...")
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(xml_files)}")
    
    # Save to JSON file
    output_path = Path(dataset_dir).parent / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted data saved to: {output_path}")
    
    # Also save as CSV for easier viewing
    csv_output = output_path.with_suffix('.csv')
    save_as_csv(all_results, csv_output)
    print(f"CSV version saved to: {csv_output}")


def save_as_csv(results: List[Dict], csv_path: Path):
    """Save results as CSV file."""
    import csv
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'PMC_ID', 'Title', 'Abstract', 'Journal', 'Year',
            'Introduction', 'Methods', 'Results', 'Discussion'
        ])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result.get('pmc_id', ''),
                result.get('title', ''),
                result.get('abstract', ''),
                result.get('metadata', {}).get('journal', ''),
                result.get('metadata', {}).get('year', ''),
                result.get('sections', {}).get('Introduction', ''),
                result.get('sections', {}).get('Methods', ''),
                result.get('sections', {}).get('Results', ''),
                result.get('sections', {}).get('Discussion', '')
            ])


if __name__ == "__main__":
    import sys
    
    # Default dataset directory
    dataset_dir = "/Users/macbookpro/Federated_Learning /PMC001xxxxxx"
    
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    output_file = "extracted_data.json"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Processing XML files from: {dataset_dir}")
    process_all_xml_files(dataset_dir, output_file)

