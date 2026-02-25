from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.lib.pagesizes import letter
from magnum_opus_chapters.utils import create_header_footer

# Import all chapters
from magnum_opus_chapters import (
    chapter_01_genesis,
    chapter_02_harvest,
    chapter_03_ledger,
    chapter_04_pulse,
    chapter_05_connection,
    chapter_06_sentinel,
    chapter_07_alchemy,
    chapter_08_experiment,
    chapter_09_prophecy,
    chapter_10_horizon,
    chapter_11_strategy,
    chapter_12_epilogue
)

def generate_pdf(output_path):
    print(f"Generating Magnum Opus to: {output_path}")
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    story = []
    
    # Generate each chapter
    print("Processing Chapter 1: The Genesis...")
    chapter_01_genesis.generate(story)
    
    print("Processing Chapter 2: The Harvest...")
    chapter_02_harvest.generate(story)
    
    print("Processing Chapter 3: The Ledger...")
    chapter_03_ledger.generate(story)
    
    print("Processing Chapter 4: The Pulse...")
    chapter_04_pulse.generate(story)
    
    print("Processing Chapter 5: The Connection...")
    chapter_05_connection.generate(story)
    
    print("Processing Chapter 6: The Sentinel...")
    chapter_06_sentinel.generate(story)
    
    print("Processing Chapter 7: The Alchemy...")
    chapter_07_alchemy.generate(story)
    
    print("Processing Chapter 8: The Experiment...")
    chapter_08_experiment.generate(story)
    
    print("Processing Chapter 9: The Prophecy...")
    chapter_09_prophecy.generate(story)
    
    print("Processing Chapter 10: The Horizon...")
    chapter_10_horizon.generate(story)
    
    print("Processing Chapter 11: The Strategy...")
    chapter_11_strategy.generate(story)
    
    print("Processing Chapter 12: The Epilogue...")
    chapter_12_epilogue.generate(story)
    
    print("Building PDF...")
    doc.build(story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
    print("Success! Magnum Opus generated.")

if __name__ == "__main__":
    generate_pdf("Stock_Market_Alchemist_Magnum_Opus.pdf")
