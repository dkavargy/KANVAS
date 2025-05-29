import pandas as pd
from tqdm import tqdm
from esco_skill_extractor import SkillExtractor

# === Step 1: Load the translated file ===
df = pd.read_csv("C:/Users/USER/PycharmProjects/esco-skill-extractor_occup/combined_linkedin_traditional_translated_en.csv")

# === Step 2: Initialize the SkillExtractor ===
skill_extractor = SkillExtractor()

# === Step 3: Extract ESCO skills from 'description_translated' ===
print("🔍 Extracting ESCO skills from translated descriptions...")
esco_skills = []

for idx, text in tqdm(enumerate(df["Descriptions_Translated"]), total=len(df), desc="Extracting skills"):
    try:
        extracted = skill_extractor.get_skills([text])
        skills_list = extracted[0]
        skills_joined = ", ".join(skills_list)
        esco_skills.append(skills_joined)

        print(f"✅ Job {idx + 1} done → Extracted: {skills_joined if skills_joined else 'No skills found'}")

    except Exception as e:
        print(f"⚠️ Error in job {idx + 1}: {e}")
        esco_skills.append("")

# === Step 4: Add new column to DataFrame ===
df["esco_skills"] = esco_skills

# === Step 5: Save result to a new file ===
output_path = "add_3k_with_esco_skills.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ All jobs processed! File saved at: {output_path}")