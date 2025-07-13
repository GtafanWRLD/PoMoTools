import os
import json
import requests
import time
import threading
import re
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import langdetect
from langdetect import detect, DetectorFactory
import requests 
from datetime import datetime
import subprocess
import polib
from shutil import copyfile

DetectorFactory.seed = 0

# ===============================================================================
# PO TRANSLATOR CONFIGURATION AND GLOBALS
# ===============================================================================

CONFIG_FILE = "config.json"
KEY_FILE = "deepseek_key.txt"

# Optimized default config for faster translations
DEFAULT_CONFIG = {
    "batch_size": 50,      # Increased from 20
    "max_workers": 12,     # Increased from 6
    "api_delay": 0.05,     # Reduced from 0.1
    "max_retries": 3,
    "min_text_length": 2,
    "max_tokens": 6000,    # Increased from 4000
    "temperature": 0.05    # Reduced for more consistent translations
}

LANGUAGE_CODES = {
    "EN": "English",
    "ES": "Spanish", 
    "FR": "French",
    "DE": "German",
}

# Language detection mapping (langdetect codes to our codes)
LANGDETECT_TO_CODE = {
    'en': 'EN',
    'es': 'ES', 
    'fr': 'FR',
    'de': 'DE',
}

COORD_MAP = {
    'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ж':'Zh','З':'Z','И':'I',
    'К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P','Р':'R','С':'S','Т':'T',
    'У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'Ch','Ш':'Sh','Щ':'Shch','Ы':'Y','Э':'E','Ю':'Yu','Я':'Ya'
}

progress_lock = threading.Lock()
api_semaphore = None

# ===============================================================================
# PO TRANSLATOR CLASSES AND FUNCTIONS
# ===============================================================================

class DeepseekClient:
    """Enhanced Deepseek AI API Client for translation"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def translate(self, texts, source_language="auto", target_language="EN"):
        """
        Enhanced translation method with better batch processing
        """
        if not texts:
            return []
        
        target_lang_name = LANGUAGE_CODES.get(target_language, target_language)
        
        # Create optimized batch format
        batch_items = []
        for i, text in enumerate(texts):
            batch_items.append(f"[{i+1}] {text}")
        
        batch_text = "\n\n".join(batch_items)
        
        # Enhanced system prompt for better performance
        system_prompt = (
            f"Translate to {target_lang_name}. CRITICAL RULES:\n"
            "1. PRESERVE ALL variables: %(var)s, %d, {0}, {{key}}, etc.\n"
            "2. PRESERVE ALL formatting: \\n, \\t, spacing, line breaks\n"
            "3. PRESERVE ALL special chars and symbols\n"
            f"4. Output natural {target_lang_name} for gaming/software content\n"
            "5. Keep [N] numbering format\n"
            "6. Military terms: танк=tank, броня=armor, урон=damage\n"
            "7. NO extra text or explanations\n"
            "8. If text has \\n keep as literal \\n characters\n\n"
            "Output format: [N] translated_text"
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": batch_text
                }
            ],
            "temperature": config.get('temperature', 0.05),
            "max_tokens": config.get('max_tokens', 6000),
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=180
        )
        
        if response.status_code != 200:
            raise Exception(f"Deepseek API error: {response.status_code} - {response.text}")
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Enhanced parsing with multiple fallback methods
        translations = self._parse_translations(content, texts)
        
        return translations
    
    def _parse_translations(self, content, original_texts):
        """Enhanced translation parsing with multiple fallback methods"""
        translations = []
        
        # Method 1: Parse [N] format
        lines = content.split('\n')
        parsed = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for [N] pattern
            match = re.match(r'\[(\d+)\]\s*(.*)', line)
            if match:
                idx = int(match.group(1)) - 1
                translation = match.group(2).strip()
                if 0 <= idx < len(original_texts):
                    parsed[idx] = translation
        
        # Method 2: Fallback - split by double newlines and match by position
        if len(parsed) < len(original_texts):
            sections = re.split(r'\n\s*\n', content)
            for i, section in enumerate(sections):
                if i < len(original_texts) and i not in parsed:
                    # Clean up the section
                    cleaned = re.sub(r'^\[\d+\]\s*', '', section.strip())
                    if cleaned:
                        parsed[i] = cleaned
        
        # Build final translations list
        for i in range(len(original_texts)):
            if i in parsed and parsed[i].strip():
                translations.append(parsed[i].strip())
            else:
                # Fallback to original
                translations.append(original_texts[i])
        
        return translations

def ensure_and_load_deepseek_key():
    """
    Ensures deepseek_key.txt exists and contains a key. Opens it in notepad if needed.
    Waits for user to paste/save, then presses Enter to continue.
    Returns the API key string.
    """
    keyfile = "deepseek_key.txt"
    while True:
        if not os.path.exists(keyfile) or not open(keyfile, encoding="utf-8").read().strip():
            # Create empty file if missing
            if not os.path.exists(keyfile):
                with open(keyfile, "w", encoding="utf-8") as f:
                    f.write("")
            print(f"\n🔑 Please paste your Deepseek API key into the file: {os.path.abspath(keyfile)}")
            print("Opening it in your default text editor...")
            # Cross-platform open
            try:
                if sys.platform.startswith('win'):
                    os.startfile(keyfile)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', keyfile])
                else:
                    subprocess.Popen(['xdg-open', keyfile])
            except Exception as e:
                print(f"⚠️  Could not open editor automatically: {e}")
            input("After pasting & saving your key, press Enter to continue...")
        # Try to read the key
        try:
            key = open(keyfile, encoding="utf-8").read().strip()
            if key:
                return key
            else:
                print("⚠️  No key detected in the file. Please paste and save your key, then press Enter.")
        except Exception as e:
            print(f"❌ Error reading {keyfile}: {e}")
            sys.exit(1)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                config = DEFAULT_CONFIG.copy()
                config.update(loaded)
                return config
        except Exception:
            print(f"❌ Invalid JSON in '{CONFIG_FILE}'. Using defaults.")
    return None

def save_config(config):
    """Save configuration to JSON file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Could not save settings: {e}")

def interactive_setup():
    print("🚀 PO Translator - First-Time Setup")
    config = DEFAULT_CONFIG.copy()
    
    print(f"\nOptimized defaults for faster translation:")
    print(f"  Batch size: {config['batch_size']} (more texts per API call)")
    print(f"  Max workers: {config['max_workers']} (more parallel requests)")
    print(f"  API delay: {config['api_delay']}s (faster rate)")
    print(f"  Max tokens: {config['max_tokens']} (larger responses)")
    
    if input("\nUse optimized defaults? (Y/n): ").lower().startswith('n'):
        try:
            config['batch_size'] = int(input(f"Batch size [{config['batch_size']}]: ") or config['batch_size'])
            config['max_workers'] = int(input(f"Max workers [{config['max_workers']}]: ") or config['max_workers'])
            config['api_delay'] = float(input(f"API delay (s) [{config['api_delay']}]: ") or config['api_delay'])
            config['max_retries'] = int(input(f"Max retries [{config['max_retries']}]: ") or config['max_retries'])
            config['min_text_length'] = int(input(f"Min text length [{config['min_text_length']}]: ") or config['min_text_length'])
        except ValueError:
            print("❌ Invalid input; using defaults.")
    
    save_config(config)
    return config

def read_po_file(path):
    try:
        import chardet
        raw = open(path, 'rb').read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        text = raw.decode(enc, errors='replace').splitlines()
        return text, enc
    except ImportError:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().splitlines()
        return text, 'utf-8'

def detect_language_enhanced(text):
    """
    Enhanced language detection with better accuracy
    Returns language code (EN, RU, PL, etc.) or 'unknown'
    """
    if not text or len(text.strip()) < 3:
        return 'unknown'
    
    try:
        # Clean text for detection - remove variables and formatting but keep more text
        clean_text = re.sub(r'%\([^)]+\)s|%\w+|{\w+}|\\[ntr]', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Remove special chars but keep letters
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 3:
            return 'unknown'
        
        # Use langdetect with confidence check
        detected = detect(clean_text)
        
        # Convert langdetect code to our format
        our_code = LANGDETECT_TO_CODE.get(detected, 'unknown')
        
        # Additional checks for common cases
        if our_code == 'unknown':
            # Check for Cyrillic (Russian/Ukrainian)
            if re.search(r'[\u0400-\u04FF]', text):
                # More specific detection between RU and UK
                if re.search(r'[іїєґ]', text):  # Ukrainian specific chars
                    return 'UK'
                else:
                    return 'RU'
            # Check for Polish specific characters
            elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                return 'PL'
            # Check for basic Latin
            elif re.search(r'[a-zA-Z]', text):
                return 'EN'  # Default to English for Latin script
        
        return our_code
        
    except Exception as e:
        # Fallback to regex-based detection
        if re.search(r'[\u0400-\u04FF]', text):
            if re.search(r'[іїєґ]', text):
                return 'UK'
            else:
                return 'RU'
        elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
            return 'PL'
        elif re.search(r'[a-zA-Z]', text):
            return 'EN'
        return 'unknown'

def should_translate_content(content, target_language):
    """
    Enhanced logic to determine if content should be translated
    Returns (should_translate: bool, reason: str, detected_lang: str)
    """
    if not content or not content.strip():
        return False, 'empty', 'unknown'
    
    # Check minimum length
    if len(content.strip()) < config.get('min_text_length', 2):
        return False, 'too_short', 'unknown'
    
    # Detect language
    detected_lang = detect_language_enhanced(content)
    
    # Check if it's coordinate format
    if is_coordinate_format(content):
        return False, 'coordinate', detected_lang
    
    # Check for patterns that shouldn't be translated
    skip_patterns = [
        r'^[A-Z]\d+$',  # Coordinate patterns
        r'^\d+$',       # Pure numbers
        r'^[.,:;!?]+$', # Pure punctuation
        r'^[A-Za-z0-9_]+\.(png|jpg|jpeg|gif|svg|wav|mp3|ogg)$',  # File names
        r'^#[0-9A-Fa-f]{6}$',  # Hex colors
        r'^\w+://',     # URLs
        r'^[A-Za-z0-9_]+$',  # Single words that might be identifiers
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, content.strip()):
            return False, 'skip_pattern', detected_lang
    
    # Main logic: translate if detected language is different from target
    if detected_lang == 'unknown':
        # If we can't detect, assume it needs translation (conservative approach)
        return True, 'unknown_assume_translate', detected_lang
    elif detected_lang == target_language:
        # Same language as target - don't translate
        return False, 'same_as_target', detected_lang
    else:
        # Different language - translate it
        return True, 'different_language', detected_lang

def is_coordinate_format(text):
    """Check if text is a coordinate format like A1, B2, etc."""
    if not text or len(text.strip()) > 10:
        return False
    
    text = text.strip()
    
    # Check for simple coordinate patterns
    if re.match(r'^[A-Z]\d+$', text):
        return True
    
    # Check for Cyrillic coordinates
    if re.match(r'^[А-Я]\d+$', text):
        return True
    
    return False

def translate_coordinate(coord_text):
    """Transliterate Cyrillic coordinates to Latin"""
    if not coord_text:
        return coord_text
    
    result = ""
    for char in coord_text:
        if char in COORD_MAP:
            result += COORD_MAP[char]
        else:
            result += char
    
    return result

def escape_po(text):
    """Properly escape text for PO format - FIXED to handle Russian quotes properly"""
    if not text:
        return ""
    
    escaped = text.replace('\\', '\\\\')  # Escape backslashes first
    escaped = escaped.replace('"', '\\"')  # Then escape ASCII double quotes only
    
    return escaped

def unescape_po(text):
    """Unescape PO format text back to original"""
    if not text:
        return ""
    
    # Reverse the escaping process
    unescaped = text.replace('\\"', '"')  # Unescape quotes first
    unescaped = unescaped.replace('\\\\', '\\')  # Then unescape backslashes
    
    return unescaped

def format_po_string(text, indent=""):
    """Format text for PO file with FIXED multiline handling"""
    if not text:
        return f'{indent}msgstr ""'
    
    # Check if text contains actual newlines (not \n literals)
    has_real_newlines = '\n' in text and not text.replace('\\n', '').find('\n') == -1
    
    if has_real_newlines:
        # Handle actual multiline content
        lines = text.split('\n')
        result = [f'{indent}msgstr ""']
        for i, line in enumerate(lines):
            escaped_line = escape_po(line)
            if i == len(lines) - 1:
                # Last line - don't add \n unless original had it
                result.append(f'{indent}"{escaped_line}"')
            else:
                result.append(f'{indent}"{escaped_line}\\n"')
        return '\n'.join(result)
    else:
        # Single line or text with \n literals - keep as is
        escaped = escape_po(text)
        return f'{indent}msgstr "{escaped}"'

def find_msgstr_blocks(lines, target_language):
    """Enhanced msgstr block detection with plural form support"""
    blocks = []
    i = 0
    idx = 0
    stats = {
        'empty': 0, 'too_short': 0, 'same_as_target': 0, 
        'coordinate': 0, 'skip_pattern': 0, 'different_language': 0,
        'unknown_assume_translate': 0
    }
    lang_stats = {}
    
    while i < len(lines):
        # Check if this is a plural form block
        has_plural = False
        if i > 0 and 'msgid_plural' in lines[i-1]:
            has_plural = True
        
        if lines[i].lstrip().startswith('msgstr') and not has_plural:
            # Regular singular msgstr
            start = i
            original = []
            line = lines[i].lstrip()
            
            # Store the original indentation
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            
            if line.strip() == 'msgstr ""':
                # Multiline msgstr
                i += 1
                while i < len(lines) and lines[i].strip().startswith('"') and lines[i].strip().endswith('"'):
                    content = lines[i].strip()[1:-1]  # Remove quotes
                    original.append(content)
                    i += 1
            else:
                # Single line msgstr
                content = line.partition(' ')[2].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]  # Remove quotes
                original.append(content)
                i += 1
            
            # Join content and unescape for analysis
            content = ''.join(original)
            # Unescape for language detection
            analysis_content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            # Enhanced content analysis
            should_translate, reason, detected_lang = should_translate_content(analysis_content, target_language)
            
            # Update statistics
            stats[reason] = stats.get(reason, 0) + 1
            if detected_lang != 'unknown':
                lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
            
            blocks.append({
                'start': start, 
                'end': i-1, 
                'idx': idx, 
                'content': content,  # Keep original escaped format
                'original_unescaped': analysis_content,  # Store unescaped for translation
                'reason': reason,
                'should_translate': should_translate,
                'detected_lang': detected_lang,
                'indent': indent,
                'is_multiline': line.strip() == 'msgstr ""',
                'is_plural': False
            })
            
            idx += 1
            
        elif lines[i].lstrip().startswith('msgstr['):
            # Plural form - handle multiple msgstr[n] entries
            start = i
            plural_forms = []
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            
            # Collect all msgstr[n] entries
            while i < len(lines) and lines[i].lstrip().startswith('msgstr['):
                line = lines[i].lstrip()
                form_num = int(line.split('[')[1].split(']')[0])
                
                original = []
                if line.strip().endswith('""'):
                    # Multiline msgstr[n]
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith('"') and lines[i].strip().endswith('"'):
                        content = lines[i].strip()[1:-1]  # Remove quotes
                        original.append(content)
                        i += 1
                else:
                    # Single line msgstr[n]
                    content = line.partition(' ')[2].strip()
                    if content.startswith('"') and content.endswith('"'):
                        content = content[1:-1]  # Remove quotes
                    original.append(content)
                    i += 1
                
                plural_forms.append({
                    'form': form_num,
                    'content': ''.join(original)
                })
            
            # Analyze the first plural form for translation decision
            if plural_forms:
                first_content = plural_forms[0]['content']
                analysis_content = first_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                should_translate, reason, detected_lang = should_translate_content(analysis_content, target_language)
                
                # Update statistics
                stats[reason] = stats.get(reason, 0) + 1
                if detected_lang != 'unknown':
                    lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
                
                blocks.append({
                    'start': start,
                    'end': i-1,
                    'idx': idx,
                    'content': first_content,  # Use first form for translation
                    'original_unescaped': analysis_content,
                    'plural_forms': plural_forms,
                    'reason': reason,
                    'should_translate': should_translate,
                    'detected_lang': detected_lang,
                    'indent': indent,
                    'is_multiline': False,  # Will be determined per form
                    'is_plural': True
                })
                
                idx += 1
        else:
            i += 1
    
    # Print enhanced statistics
    total_blocks = len(blocks)
    translatable = sum(1 for b in blocks if b['should_translate'])
    
    print(f"📊 Content Analysis (Target: {LANGUAGE_CODES.get(target_language, target_language)}):")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Will translate: {translatable}")
    print(f"   Will skip: {total_blocks - translatable}")
    
    print(f"\n📈 Detected languages:")
    for lang, count in sorted(lang_stats.items(), key=lambda x: x[1], reverse=True):
        lang_name = LANGUAGE_CODES.get(lang, lang)
        print(f"   {lang_name}: {count}")
    
    print(f"\n📋 Skip reasons:")
    for reason, count in stats.items():
        if count > 0 and reason != 'different_language' and reason != 'unknown_assume_translate':
            print(f"   {reason.replace('_', ' ').title()}: {count}")
    
    return blocks

def translate_batch(batch, batch_id, progress, target_language):
    """Enhanced batch translation with better language handling"""
    if not batch:
        return {}
    
    # Separate content by type
    coords = [b for b in batch if b['reason'] == 'coordinate']
    translatable = [b for b in batch if b['should_translate']]
    skippable = [b for b in batch if not b['should_translate']]
    
    results = {}
    
    # Handle coordinates (no API call)
    for b in coords:
        results[b['idx']] = translate_coordinate(b['content'])
    
    # Skip content that doesn't need translation (no API call)
    # IMPORTANT: Keep original content as-is without re-escaping
    for b in skippable:
        results[b['idx']] = b['content']  # Keep original escaped format
    
    # Only make API call if there's translatable content
    if translatable:
        with api_semaphore:
            time.sleep(config['api_delay'])
            # Use unescaped content for translation
            texts = [b['original_unescaped'] for b in translatable]
            
            for attempt in range(config['max_retries'] + 1):
                try:
                    translations = client.translate(
                        texts=texts,
                        source_language='auto',
                        target_language=target_language
                    )
                    
                    if len(translations) != len(texts):
                        raise Exception(f"Translation count mismatch: got {len(translations)}, expected {len(texts)}")
                    
                    for i, translation in enumerate(translations):
                        if translation and translation.strip():
                            # Clean up any contamination in translation
                            cleaned = translation.strip()
                            
                            # Check for contamination (paths that shouldn't be there)
                            original = texts[i]
                            if ('buyingPanel/' in cleaned or 'infoPanel/' in cleaned) and not ('buyingPanel/' in original or 'infoPanel/' in original):
                                print(f"⚠️  Detected contaminated translation, using original")
                                results[translatable[i]['idx']] = translatable[i]['content']
                            else:
                                # Escape the translated content for PO format
                                escaped_translation = escape_po(cleaned)
                                results[translatable[i]['idx']] = escaped_translation
                        else:
                            results[translatable[i]['idx']] = translatable[i]['content']
                    
                    break
                    
                except Exception as e:
                    if attempt < config['max_retries']:
                        wait_time = (attempt + 1) * 2
                        print(f"⚠️  Batch {batch_id} attempt {attempt + 1} failed: {e}")
                        print(f"   Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Batch {batch_id} failed after {config['max_retries']} retries: {e}")
                        for b in translatable:
                            results[b['idx']] = b['content']
    
    # Update progress for ALL items in batch, not just translated ones
    with progress_lock:
        progress.update(len(batch))
    
    return results

def process_file(src, dst, target_language):
    print(f"Processing {src}...")
    lines, enc = read_po_file(src)
    blocks = find_msgstr_blocks(lines, target_language)
    total = len(blocks)
    
    if total == 0:
        print(f"No msgstr blocks found in {src}")
        return 0
    
    # Create batches
    batches = [blocks[i:i + config['batch_size']] for i in range(0, total, config['batch_size'])]
    progress = tqdm(total=total, desc='Translating', unit='str')
    
    all_results = {}
    
    # Process batches with threading
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = {executor.submit(translate_batch, batch, idx, progress, target_language): idx
                   for idx, batch in enumerate(batches)}
        
        for fut in futures:
            try:
                batch_results = fut.result()
                all_results.update(batch_results)
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
    
    progress.close()
    
    # Reconstruct the PO file
    out = []
    cur = 0
    
    for b in blocks:
        # Copy lines before this msgstr block
        out.extend(lines[cur:b['start']])
        
        # Get translation (already properly escaped if it was translated)
        trans = all_results.get(b['idx'], b['content'])
        indent = lines[b['start']].split('msgstr')[0]
        
        if b['is_plural']:
            # Handle plural forms
            for form_data in b['plural_forms']:
                form_num = form_data['form']
                # Use the same translation for all plural forms for now
                # In a more sophisticated version, you might want to handle different plural forms
                form_content = trans if form_num == 0 else trans
                
                # Check if this form was multiline originally
                original_line = None
                for line_idx in range(b['start'], b['end'] + 1):
                    if f'msgstr[{form_num}]' in lines[line_idx]:
                        original_line = lines[line_idx]
                        break
                
                is_multiline = original_line and original_line.strip().endswith('""')
                
                if is_multiline or '\n' in form_content.replace('\\n', ''):
                    # Multiline format
                    out.append(f"{indent}msgstr[{form_num}] \"\"")
                    
                    # Split by literal \n in the string
                    if '\\n' in form_content:
                        parts = form_content.split('\\n')
                        for j, part in enumerate(parts):
                            if j == len(parts) - 1 and not part:
                                # Last empty part from trailing \n
                                continue
                            if j == len(parts) - 1:
                                # Last part without \n - content is already escaped
                                out.append(f"{indent}\"{part}\"")
                            else:
                                # Part with \n - content is already escaped
                                out.append(f"{indent}\"{part}\\n\"")
                    else:
                        # No \n sequences, treat as single line in multiline format
                        out.append(f"{indent}\"{form_content}\"")
                else:
                    # Single line format - content is already escaped
                    out.append(f"{indent}msgstr[{form_num}] \"{form_content}\"")
        else:
            # Handle regular singular msgstr (existing logic)
            if b['is_multiline'] or '\n' in trans.replace('\\n', ''):
                # Multiline format
                out.append(f"{indent}msgstr \"\"")
                
                # Split by literal \n in the string
                if '\\n' in trans:
                    parts = trans.split('\\n')
                    for j, part in enumerate(parts):
                        if j == len(parts) - 1 and not part:
                            # Last empty part from trailing \n
                            continue
                        if j == len(parts) - 1:
                            # Last part without \n - content is already escaped
                            out.append(f"{indent}\"{part}\"")
                        else:
                            # Part with \n - content is already escaped
                            out.append(f"{indent}\"{part}\\n\"")
                else:
                    # No \n sequences, treat as single line in multiline format
                    out.append(f"{indent}\"{trans}\"")
            else:
                # Single line format - content is already escaped
                out.append(f"{indent}msgstr \"{trans}\"")
        
        cur = b['end'] + 1
    
    # Copy remaining lines
    out.extend(lines[cur:])
    
    # Write output
    try:
        with open(dst, 'w', encoding=enc) as f:
            f.write("\n".join(out))
    except Exception as e:
        print(f"❌ Error writing {dst}: {e}")
        return 0
    
    return len([b for b in blocks if b['should_translate']])

def run_translator():
    """Run the PO Translator tool"""
    global config, client, api_semaphore
    
    print("=" * 60)
    print("                PO TRANSLATOR")
    print("=" * 60)
    
    # Create directories
    for d in ('input', 'output'):
        os.makedirs(d, exist_ok=True)
    
    # Load or create config
    config = load_config() or interactive_setup()
    
    # Language selection
    print(f"\nAvailable target languages:")
    for i, (code, name) in enumerate(LANGUAGE_CODES.items(), 1):
        print(f"  {i:2d}. {code} — {name}")
    
    while True:
        sel = input("\nEnter language code or number: ").strip().upper()
        if sel.isdigit() and 1 <= int(sel) <= len(LANGUAGE_CODES):
            target_language = list(LANGUAGE_CODES.keys())[int(sel) - 1]
            break
        if sel in LANGUAGE_CODES:
            target_language = sel
            break
        print("❌ Invalid choice. Try again.")
    
    print(f"✅ Target language: {LANGUAGE_CODES[target_language]}")
    print(f"🧠 Smart translation: Only translates text that's NOT already in {LANGUAGE_CODES[target_language]}")
    
    # Get API key
    key = os.getenv('DEEPSEEK_API_KEY')
    if not key:
        key = ensure_and_load_deepseek_key()
    
    if not key and os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, 'r', encoding='utf-8') as f:
                key = f.read().strip()
            print(f"✅ API key loaded from {KEY_FILE}")
        except Exception as e:
            print(f"❌ Error reading {KEY_FILE}: {e}")
    elif key:
        print("✅ API key loaded from environment variable")
    
    if not key:
        print(f"❌ Missing Deepseek API key.")
        print(f"   Option 1: Set environment variable: export DEEPSEEK_API_KEY='your-key'")
        print(f"   Option 2: Create file '{KEY_FILE}' with your API key")
        sys.exit(1)
    
    # Initialize client
    client = DeepseekClient(api_key=key)
    api_semaphore = threading.Semaphore(config['max_workers'])
    
    # Find PO files
    po_files = [f for f in os.listdir('input') if f.endswith('.po')]
    if not po_files:
        print("❌ No .po files found in 'input' directory.")
        return
    
    print(f"\n📁 Found {len(po_files)} .po file(s) to process")
    print(f"⚡ Performance config: {config['batch_size']} batch size, {config['max_workers']} workers, {config['api_delay']}s delay")
    
    # Process files
    total_translated = 0
    start_time = time.time()
    
    for fname in po_files:
        in_path = os.path.join('input', fname)
        out_path = os.path.join('output', fname)
        
        try:
            count = process_file(in_path, out_path, target_language)
            print(f"✅ {fname}: {count} strings translated\n")
            total_translated += count
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}\n")
    
    elapsed = time.time() - start_time
    rate = total_translated / elapsed if elapsed > 0 else 0
    print(f"🎉 Done! Translated {total_translated} strings in {elapsed:.1f}s ({rate:.1f} str/s)")

# ===============================================================================
# PO/MO CONVERTER FUNCTIONS
# ===============================================================================

def mo_to_po(mo_path, po_path):
    """Convert a single .mo file to .po file"""
    try:
        mo_entries = polib.mofile(mo_path)
        po = polib.POFile()
        for entry in mo_entries:
            po.append(entry)
        po.metadata = mo_entries.metadata
        po.save(po_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to decompile '{mo_path}': {e}")
        return False

def po_to_mo(po_path, mo_path):
    """Convert a single .po file to .mo file"""
    try:
        po = polib.pofile(po_path)
        po.save_as_mofile(mo_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to compile '{po_path}': {e}")
        return False

def decompile_folder(mo_folder, po_folder):
    """Decompile all .mo files in a folder to .po files"""
    if not os.path.exists(po_folder):
        os.makedirs(po_folder)
    
    mo_files = [f for f in os.listdir(mo_folder) if f.endswith('.mo')]
    if not mo_files:
        print("No .mo files found in the specified folder.")
        return
    
    print(f"Found {len(mo_files)} .mo files to decompile...")
    success_count = 0
    
    for filename in mo_files:
        mo_path = os.path.join(mo_folder, filename)
        po_filename = os.path.splitext(filename)[0] + '.po'
        po_path = os.path.join(po_folder, po_filename)
        print(f"Decompiling: {filename} → {po_filename} ... ", end="", flush=True)
        if mo_to_po(mo_path, po_path):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
    
    print(f"\nDecompilation complete: {success_count}/{len(mo_files)} files processed successfully.")

def compile_folder(po_folder, mo_folder):
    """Compile all .po files in a folder to .mo files"""
    if not os.path.exists(mo_folder):
        os.makedirs(mo_folder)
    
    po_files = [f for f in os.listdir(po_folder) if f.endswith('.po')]
    if not po_files:
        print("No .po files found in the specified folder.")
        return
    
    print(f"Found {len(po_files)} .po files to compile...")
    success_count = 0
    
    for filename in po_files:
        po_path = os.path.join(po_folder, filename)
        mo_filename = os.path.splitext(filename)[0] + '.mo'
        mo_path = os.path.join(mo_folder, mo_filename)
        print(f"Compiling: {filename} → {mo_filename} ... ", end="", flush=True)
        if po_to_mo(po_path, mo_path):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
    
    print(f"\nCompilation complete: {success_count}/{len(po_files)} files processed successfully.")

def run_converter():
    """Run the PO/MO converter tool"""
    print("=" * 60)
    print("           PO/MO FILE CONVERTER TOOL")
    print("=" * 60)
    
    # Choose action first
    print("\nWhat would you like to do?")
    print("1. Decompile .mo files to .po files")
    print("2. Compile .po files to .mo files")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2.")
    
    action = 'decompile' if choice == '1' else 'compile'
    file_extension = '.mo' if action == 'decompile' else '.po'
    
    # Get input folder
    while True:
        folder_path = input(f"\nEnter the folder path with your {file_extension} files: ").strip().strip('"')
        if os.path.isdir(folder_path):
            break
        print("Error: Folder does not exist. Please try again.")
    
    # Get script directory (where the .py/.exe file is located)
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # Check if files exist
    if action == 'decompile':
        mo_count = len([f for f in os.listdir(folder_path) if f.endswith('.mo')])
        if mo_count == 0:
            print("\nNo .mo files found in the specified folder.")
            return
        print(f"\nFound {mo_count} .mo files to decompile.")
    else:
        po_count = len([f for f in os.listdir(folder_path) if f.endswith('.po')])
        if po_count == 0:
            print("\nNo .po files found in the specified folder.")
            return
        print(f"\nFound {po_count} .po files to compile.")
    
    # Perform the action
    print(f"\nStarting {action}...")
    
    if action == 'decompile':
        output_folder = os.path.join(script_dir, 'decompiled')
        decompile_folder(folder_path, output_folder)
        print(f"\nDecompiled .po files are saved in: {output_folder}")
    else:
        output_folder = os.path.join(script_dir, 'compiled')
        compile_folder(folder_path, output_folder)
        print(f"\nCompiled .mo files are saved in: {output_folder}")
    
    print("\nConverter task completed!")

# ===============================================================================
# PO FILE MERGER FUNCTIONS
# ===============================================================================

def detect_po_language(po_file, sample_size=20):
    """Detect language of PO file by sampling msgstr entries"""
    try:
        po = polib.pofile(po_file)
        # Get all non-empty translated strings (including plurals)
        texts = []
        for entry in po:
            if entry.msgstr and entry.msgstr.strip():
                texts.append(entry.msgstr)
            elif hasattr(entry, 'msgstr_plural') and entry.msgstr_plural:
                # Add plural forms
                for plural_form in entry.msgstr_plural.values():
                    if plural_form and plural_form.strip():
                        texts.append(plural_form)
        
        if not texts:
            return 'empty'
        
        # Use more samples for better detection
        sample_texts = texts[:sample_size] if len(texts) >= sample_size else texts
        
        # Combine all sample texts into one for better detection
        combined_text = ' '.join(sample_texts)
        
        if len(combined_text.strip()) < 10:  # Not enough text
            return 'insufficient'
            
        try:
            detected = detect(combined_text)
            return detected
        except:
            # Fallback: try individual texts
            langs = []
            for text in sample_texts[:10]:  # Try first 10 individually
                try:
                    detected = detect(text)
                    langs.append(detected)
                except:
                    continue
            
            if langs:
                return max(set(langs), key=langs.count)
            return 'unknown'
            
    except Exception as e:
        print(f"Error detecting language for {po_file}: {e}")
        return 'error'

def get_language_name(lang_code):
    """Convert language code to readable name"""
    lang_names = {
        'en': 'English',
        'ru': 'Russian',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'zh': 'Chinese',
        'nl': 'Dutch',
        'pl': 'Polish',
    }
    return lang_names.get(lang_code, f'Unknown ({lang_code})')

def analyze_po_file(po_path):
    """Analyze PO file content for better understanding"""
    try:
        po = polib.pofile(po_path)
        total_entries = len(po)
        translated_entries = 0
        plural_entries = 0
        
        # Count translated entries (including plurals)
        for entry in po:
            if entry.msgstr and entry.msgstr.strip():
                translated_entries += 1
            elif hasattr(entry, 'msgstr_plural') and entry.msgstr_plural:
                # Check if any plural form is translated
                if any(plural_form.strip() for plural_form in entry.msgstr_plural.values()):
                    translated_entries += 1
                    plural_entries += 1
        
        empty_entries = total_entries - translated_entries
        
        # Get sample translated text for display
        sample_texts = []
        for entry in po:
            if entry.msgstr and entry.msgstr.strip():
                sample_texts.append(entry.msgstr)
            elif hasattr(entry, 'msgstr_plural') and entry.msgstr_plural:
                for plural_form in entry.msgstr_plural.values():
                    if plural_form and plural_form.strip():
                        sample_texts.append(plural_form)
                        break
            if len(sample_texts) >= 3:
                break
        
        return {
            'total': total_entries,
            'translated': translated_entries,
            'empty': empty_entries,
            'plural_entries': plural_entries,
            'samples': sample_texts
        }
    except Exception as e:
        return {'error': str(e)}

def has_translation_content(entry):
    """Check if entry has any translation content (singular or plural)"""
    if entry.msgstr and entry.msgstr.strip():
        return True
    if hasattr(entry, 'msgstr_plural') and entry.msgstr_plural:
        return any(plural_form.strip() for plural_form in entry.msgstr_plural.values())
    return False

def copy_translation_content(source_entry, target_entry):
    """Copy translation content from source to target entry (handles both singular and plural)"""
    # Copy singular translation
    if source_entry.msgstr and source_entry.msgstr.strip():
        target_entry.msgstr = source_entry.msgstr
    
    # Copy plural translations
    if hasattr(source_entry, 'msgstr_plural') and source_entry.msgstr_plural:
        # Ensure target has msgstr_plural dictionary
        if not hasattr(target_entry, 'msgstr_plural') or not target_entry.msgstr_plural:
            target_entry.msgstr_plural = {}
        
        # Copy all plural forms
        for key, value in source_entry.msgstr_plural.items():
            if value and value.strip():  # Only copy non-empty plural forms
                target_entry.msgstr_plural[key] = value

def merge_po_files(ru_path, overlay_path, merged_path):
    """Merge overlay language translations onto Russian base file"""
    try:
        # Load both PO files
        ru_po = polib.pofile(ru_path)
        overlay_po = polib.pofile(overlay_path)
        
        # Create mapping of msgid -> entry from overlay file
        overlay_map = {}
        for entry in overlay_po:
            if entry.msgid and has_translation_content(entry):
                overlay_map[entry.msgid] = entry
        
        # Stats tracking
        overlaid = 0
        kept_russian = 0
        empty_entries = 0
        plural_overlaid = 0
        total_entries = len(ru_po)
        
        # Process each entry in Russian file (preserve ALL Russian entries)
        for entry in ru_po:
            if not has_translation_content(entry):
                # Entry is empty in Russian file
                if entry.msgid in overlay_map:
                    # Fill empty Russian entry with overlay translation
                    copy_translation_content(overlay_map[entry.msgid], entry)
                    overlaid += 1
                    if hasattr(overlay_map[entry.msgid], 'msgstr_plural') and overlay_map[entry.msgid].msgstr_plural:
                        plural_overlaid += 1
                else:
                    # Keep empty
                    empty_entries += 1
            else:
                # Russian entry has content
                if entry.msgid in overlay_map:
                    # Overlay translation over Russian
                    copy_translation_content(overlay_map[entry.msgid], entry)
                    overlaid += 1
                    if hasattr(overlay_map[entry.msgid], 'msgstr_plural') and overlay_map[entry.msgid].msgstr_plural:
                        plural_overlaid += 1
                else:
                    # Keep original Russian translation
                    kept_russian += 1
        
        # Save merged file
        ru_po.save(merged_path)
        
        # Verify entry count (should be exactly the same as Russian original)
        merged_po = polib.pofile(merged_path)
        merged_count = len(merged_po)
        
        if merged_count != total_entries:
            print(f"⚠️ CRITICAL ERROR: Entry count mismatch! Russian: {total_entries}, Merged: {merged_count}")
            return {'success': False, 'error': f'Entry count mismatch: {total_entries} -> {merged_count}'}
        
        return {
            'total': total_entries,
            'overlaid': overlaid,
            'kept_russian': kept_russian,
            'empty_entries': empty_entries,
            'plural_overlaid': plural_overlaid,
            'merged_count': merged_count,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error merging {ru_path} and {overlay_path}: {e}")
        return {'success': False, 'error': str(e)}

def copy_overlay_file(overlay_path, merged_path):
    """Copy overlay file as-is when no Russian base exists"""
    try:
        copyfile(overlay_path, merged_path)
        overlay_po = polib.pofile(overlay_path)
        return {
            'total': len(overlay_po),
            'copied_overlay': True,
            'success': True
        }
    except Exception as e:
        print(f"❌ Error copying {overlay_path}: {e}")
        return {'success': False, 'error': str(e)}

def copy_russian_file(ru_path, merged_path):
    """Copy Russian file as-is when no overlay version exists"""
    try:
        copyfile(ru_path, merged_path)
        ru_po = polib.pofile(ru_path)
        return {
            'total': len(ru_po),
            'copied': True,
            'success': True
        }
    except Exception as e:
        print(f"❌ Error copying {ru_path}: {e}")
        return {'success': False, 'error': str(e)}

def get_matching_files(ru_dir, overlay_dir):
    """Get matching files between Russian and overlay directories"""
    ru_files = {f for f in os.listdir(ru_dir) if f.endswith('.po')} if os.path.exists(ru_dir) else set()
    overlay_files = {f for f in os.listdir(overlay_dir) if f.endswith('.po')} if os.path.exists(overlay_dir) else set()
    
    # Files that exist in both directories (exact name match)
    matching_files = ru_files & overlay_files
    
    # Files that exist only in Russian
    ru_only_files = ru_files - overlay_files
    
    # Files that exist only in overlay language
    overlay_only_files = overlay_files - ru_files
    
    return {
        'ru_files': ru_files,
        'overlay_files': overlay_files,
        'matching': matching_files,
        'ru_only': ru_only_files,
        'overlay_only': overlay_only_files
    }

def setup_merger_directories():
    """Setup and explain directory structure for merger"""
    print("🚀 UNIVERSAL PO FILE MERGER (with Plural Support)")
    print("=" * 60)
    print("This tool merges translation files, overlaying any language onto Russian base files.")
    print("✅ Now supports plural forms (msgstr[0], msgstr[1], msgstr[2], etc.)")
    print("✅ Copies overlay-only files (files without Russian base)")
    print("🚫 Russian + Russian merging is automatically blocked.")
    print()
    
    # Create directories
    ru_dir = "ru"
    merged_dir = "merged"
    
    os.makedirs(ru_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    
    print("📁 FOLDER SETUP:")
    print(f"✅ '{ru_dir}' folder created/exists - PUT YOUR RUSSIAN .po FILES HERE")
    print(f"✅ '{merged_dir}' folder created/exists - merged files will be saved here")
    print()
    
    return ru_dir, merged_dir

def get_overlay_directory():
    """Get overlay directory from user input"""
    while True:
        print("🌍 OVERLAY LANGUAGE SETUP:")
        print("You need to specify where your overlay language files are located.")
        print()
        overlay_dir = input("Enter the folder name for your overlay language files (e.g., 'english', 'spanish', 'polish', 'input'): ").strip()
        
        if not overlay_dir:
            print("❌ Please enter a folder name.")
            continue
            
        if overlay_dir.lower() in ['ru', 'russian', 'merged']:
            print("❌ This folder name is reserved. Please choose a different name.")
            continue
            
        # Create the directory
        os.makedirs(overlay_dir, exist_ok=True)
        print(f"✅ '{overlay_dir}' folder created/ready - PUT YOUR OVERLAY LANGUAGE .po FILES HERE")
        print()
        
        return overlay_dir

def run_merger():
    """Run the PO file merger tool"""
    print("=" * 60)
    print("               PO FILE MERGER")
    print("=" * 60)
    
    # Setup directories
    ru_dir, merged_dir = setup_merger_directories()
    overlay_dir = get_overlay_directory()
    
    # Check if Russian files exist
    if not os.path.exists(ru_dir) or not any(f.endswith('.po') for f in os.listdir(ru_dir)):
        print(f"❌ No .po files found in '{ru_dir}' folder.")
        print(f"📁 Please place your Russian .po files in the '{ru_dir}' folder and run again.")
        return
    
    # Check if overlay files exist
    if not os.path.exists(overlay_dir) or not any(f.endswith('.po') for f in os.listdir(overlay_dir)):
        print(f"⚠️ No .po files found in '{overlay_dir}' folder.")
        print(f"📁 Please place your overlay language .po files in the '{overlay_dir}' folder.")
        
        choice = input("Continue with Russian-only files? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting...")
            return
    
    print("🔄 ANALYZING FILES...")
    print("=" * 60)
    
    # Get file matching information
    file_info = get_matching_files(ru_dir, overlay_dir)
    
    print(f"📊 FILE ANALYSIS:")
    print(f"   🇷🇺 Russian files: {len(file_info['ru_files'])}")
    print(f"   🌍 Overlay files: {len(file_info['overlay_files'])}")
    print(f"   🔗 Matching names: {len(file_info['matching'])}")
    print(f"   🇷🇺 Russian only: {len(file_info['ru_only'])}")
    print(f"   🌍 Overlay only: {len(file_info['overlay_only'])}")
    print()
    
    # Statistics
    total_stats = {
        'files_processed': 0,
        'files_merged': 0,
        'files_copied': 0,
        'overlay_only_copied': 0,
        'files_skipped': 0,
        'total_entries': 0,
        'total_overlaid': 0,
        'total_kept_russian': 0,
        'total_empty': 0,
        'total_plural_overlaid': 0
    }
    
    # Process files with matching names (merge with language detection)
    if file_info['matching']:
        print("🔍 PROCESSING MATCHING FILES:")
        print("-" * 40)
        
        for ru_filename in sorted(file_info['matching']):
            ru_path = os.path.join(ru_dir, ru_filename)
            overlay_path = os.path.join(overlay_dir, ru_filename)
            merged_path = os.path.join(merged_dir, ru_filename)
            
            print(f"\n📄 {ru_filename}")
            
            # Analyze files
            ru_analysis = analyze_po_file(ru_path)
            overlay_analysis = analyze_po_file(overlay_path)
            
            print(f"   🇷🇺 Russian file: {ru_analysis['translated']}/{ru_analysis['total']} translated")
            if ru_analysis.get('plural_entries', 0) > 0:
                print(f"      📝 Plural entries: {ru_analysis['plural_entries']}")
            if ru_analysis['samples']:
                print(f"      Sample: \"{ru_analysis['samples'][0][:50]}{'...' if len(ru_analysis['samples'][0]) > 50 else ''}\"")
            
            print(f"   🌍 Overlay file: {overlay_analysis['translated']}/{overlay_analysis['total']} translated")
            if overlay_analysis.get('plural_entries', 0) > 0:
                print(f"      📝 Plural entries: {overlay_analysis['plural_entries']}")
            if overlay_analysis['samples']:
                print(f"      Sample: \"{overlay_analysis['samples'][0][:50]}{'...' if len(overlay_analysis['samples'][0]) > 50 else ''}\"")
            
            # Language detection
            print(f"   🔍 Detecting languages...")
            ru_lang = detect_po_language(ru_path)
            overlay_lang = detect_po_language(overlay_path)
            
            ru_lang_name = get_language_name(ru_lang)
            overlay_lang_name = get_language_name(overlay_lang)
            
            print(f"   🇷🇺 Russian file detected as: {ru_lang_name}")
            print(f"   🌍 Overlay file detected as: {overlay_lang_name}")
            
            # Check for Russian-on-Russian merge
            if overlay_lang == 'ru':
                print(f"   🚫 SKIPPING: Both files are Russian - preventing Russian+Russian merge")
                print(f"   📄 Copying Russian file as-is...")
                result = copy_russian_file(ru_path, merged_path)
                
                if result['success']:
                    total_stats['files_skipped'] += 1
                    total_stats['total_entries'] += result['total']
                    print(f"   ✅ Russian file copied! ({result['total']} entries)")
                else:
                    print(f"   ❌ Copy failed: {result.get('error', 'Unknown error')}")
            else:
                # Proceed with merge
                if overlay_lang in ['empty', 'unknown', 'error', 'insufficient']:
                    print(f"   ⚠️ Warning: Overlay language detection: {overlay_lang}")
                
                print(f"   📝 Merging {overlay_lang_name} → Russian...")
                result = merge_po_files(ru_path, overlay_path, merged_path)
                
                if result['success']:
                    total_stats['files_merged'] += 1
                    total_stats['total_entries'] += result['total']
                    total_stats['total_overlaid'] += result['overlaid']
                    total_stats['total_kept_russian'] += result['kept_russian']
                    total_stats['total_empty'] += result['empty_entries']
                    total_stats['total_plural_overlaid'] += result.get('plural_overlaid', 0)
                    
                    print(f"   ✅ Merged successfully!")
                    print(f"      📊 Total: {result['total']} | Overlaid: {result['overlaid']} | Kept Russian: {result['kept_russian']} | Empty: {result['empty_entries']}")
                    if result.get('plural_overlaid', 0) > 0:
                        print(f"      📝 Plural forms overlaid: {result['plural_overlaid']}")
                else:
                    print(f"   ❌ Merge failed: {result.get('error', 'Unknown error')}")
            
            total_stats['files_processed'] += 1
    
    # Process Russian-only files (copy)
    if file_info['ru_only']:
        print(f"\n📄 PROCESSING RUSSIAN-ONLY FILES:")
        print("-" * 40)
        
        for ru_filename in sorted(file_info['ru_only']):
            ru_path = os.path.join(ru_dir, ru_filename)
            merged_path = os.path.join(merged_dir, ru_filename)
            
            print(f"\n📄 {ru_filename}")
            print(f"   📄 No overlay version found, copying Russian file as-is...")
            result = copy_russian_file(ru_path, merged_path)
            
            if result['success']:
                total_stats['files_copied'] += 1
                total_stats['total_entries'] += result['total']
                print(f"   ✅ Copied! ({result['total']} entries)")
            else:
                print(f"   ❌ Copy failed: {result.get('error', 'Unknown error')}")
            
            total_stats['files_processed'] += 1
    
    # Process overlay-only files (copy these as well)
    if file_info['overlay_only']:
        print(f"\n🌍 PROCESSING OVERLAY-ONLY FILES:")
        print("-" * 40)
        
        for overlay_filename in sorted(file_info['overlay_only']):
            overlay_path = os.path.join(overlay_dir, overlay_filename)
            merged_path = os.path.join(merged_dir, overlay_filename)
            
            print(f"\n📄 {overlay_filename}")
            
            # Analyze the overlay file
            overlay_analysis = analyze_po_file(overlay_path)
            print(f"   🌍 Overlay file: {overlay_analysis['translated']}/{overlay_analysis['total']} translated")
            if overlay_analysis.get('plural_entries', 0) > 0:
                print(f"      📝 Plural entries: {overlay_analysis['plural_entries']}")
            if overlay_analysis['samples']:
                print(f"      Sample: \"{overlay_analysis['samples'][0][:50]}{'...' if len(overlay_analysis['samples'][0]) > 50 else ''}\"")
            
            # Language detection
            overlay_lang = detect_po_language(overlay_path)
            overlay_lang_name = get_language_name(overlay_lang)
            print(f"   🔍 Detected as: {overlay_lang_name}")
            
            print(f"   📄 No Russian base found, copying overlay file as-is...")
            result = copy_overlay_file(overlay_path, merged_path)
            
            if result['success']:
                total_stats['overlay_only_copied'] += 1
                total_stats['total_entries'] += result['total']
                print(f"   ✅ Copied! ({result['total']} entries)")
            else:
                print(f"   ❌ Copy failed: {result.get('error', 'Unknown error')}")
            
            total_stats['files_processed'] += 1
    
    # Final summary
    print(f"\n🎯 MERGE COMPLETE!")
    print("=" * 60)
    print(f"📁 Files processed: {total_stats['files_processed']}")
    print(f"🔄 Files merged: {total_stats['files_merged']}")
    print(f"📄 Russian files copied: {total_stats['files_copied']}")
    print(f"🌍 Overlay-only files copied: {total_stats['overlay_only_copied']}")
    print(f"🚫 Files skipped (RU+RU): {total_stats['files_skipped']}")
    print(f"📊 Total entries: {total_stats['total_entries']}")
    print(f"🔄 Overlaid entries: {total_stats['total_overlaid']}")
    print(f"📝 Plural forms overlaid: {total_stats['total_plural_overlaid']}")
    print(f"🇷🇺 Kept Russian: {total_stats['total_kept_russian']}")
    print(f"⭕ Empty entries: {total_stats['total_empty']}")
    print(f"📁 Results saved in '{merged_dir}' folder")
    
    print(f"\n✅ Done! Check the '{merged_dir}' folder for your merged files.")
    print(f"📊 Total files in merged folder: {total_stats['files_merged'] + total_stats['files_copied'] + total_stats['overlay_only_copied'] + total_stats['files_skipped']}")

# ===============================================================================
# MAIN PROGRAM
# ===============================================================================

def main():
    """Main program - choose between translator, converter, and merger"""
    print("=" * 70)
    print("                    UNIVERSAL PO/MO TOOLKIT")
    print("=" * 70)
    print("Choose which tool you want to use:")
    print()
    print("1. PO Translator (AI-powered translation with Deepseek)")
    print("   - Translate .po files using AI")
    print("   - Smart language detection and filtering")
    print("   - Supports multiple target languages")
    print()
    print("2. PO/MO File Converter")
    print("   - Convert .mo files to .po files (decompile)")
    print("   - Convert .po files to .mo files (compile)")
    print()
    print("3. PO File Merger")
    print("   - Merge translation files (overlay any language onto Russian base)")
    print("   - Supports plural forms and language detection")
    print("   - Prevents Russian+Russian merging")
    print()
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3.")
    
    print()
    
    if choice == '1':
        run_translator()
    elif choice == '2':
        run_converter()
    else:
        run_merger()

if __name__ == "__main__":
    try:
        main()
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)