#  PoMoMultiTool

A universal tool designed for working with **Lesta (ex-Wargaming)** `.po` and `.mo` language files used in games like Mir Tankov, Mir Korabley and similar.

---

##  What It Does

 **1. AI-Powered `.po` Translator**  
- Translates `.po` files using **Deepseek AI**.
- Automatically detects source language and only translates strings that are not already in the target language.
- Smart handling for variables, coordinates, plural forms, and gaming-specific terms.
- Designed for bulk processing of Lesta `.po` files while preserving original file formatting.

**2. `.po` ⇄ `.mo` File Converter**  
- Converts `.mo` files to `.po` (decompile).
- Compiles `.po` files back to `.mo`.
- Perfect for editing language files and repacking them for Lesta-based game engines.

 **3. `.po` File Merger (with Plural Support)**  
- Merges overlay translations onto a base Russian `.po` file.
- Supports plural forms (`msgstr[0]`, `msgstr[1]`, etc.) to ensure no data loss.
- Prevents accidental merging of Russian onto Russian files.
- Ideal for maintaining multiple language overlays for the same mod or game update.

---

## **Who It’s For**

This tool is meant for:
- **Modders**, localizers, and fans working with **Lesta/Wargaming `.po` and `.mo` files**.
- Anyone needing to translate or merge large language file sets.

---



##  **Recommended Project Structure**

- `input/` – Put your `.po` files here for translation.
- `output/` – Translated files will be saved here.
- `ru/` – Russian base `.po` files for merging.
- `merged/` – Result of merged overlay files.
- `deepseek_key.txt` – Your Deepseek API key for translation.

---

## **How To Use**

1. Put your `.po` files in the relevant folders (`input/`, `ru/`, etc.).
2. To run the source code run:
   ```bash
   python PoMoMultiTool.py

   ```
OR

Run the .exe from the releases.
