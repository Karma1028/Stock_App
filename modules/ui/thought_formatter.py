import re

def parse_and_format_thought(raw_text):
    if not raw_text:
        return ""
        
    # Strip any potential CDATA wrappers an LLM might generate
    raw_text = raw_text.replace("<!Cdata[", "").replace("<![CDATA[", "").replace("]]>", "").replace("]>", "")
        
    lines = raw_text.split('\n')
    
    html_out = """<div style="background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #1e293b; font-family: 'Inter', sans-serif;">"""
    in_section = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if in_section:
                html_out += '<div style="height: 6px;"></div>'
            continue
            
        is_header = False
        clean_title = line
        
        # Stricter heuristics to identify a "step" or "header", explicitly ignoring JSON or code chars
        has_code_chars = any(c in line for c in ['{', '}', '"', '[', ']', '<', '>'])
        
        if not has_code_chars and len(line) >= 3:
            if line.startswith('#'):
                is_header = True
                clean_title = line.lstrip('#').strip().title()
            elif line.endswith(':'):
                is_header = True
                clean_title = line[:-1].strip().title()
            elif any(line.lower().startswith(kw) for kw in ['step ', 'phase ', 'analyzing', 'calculating', 'evaluating', 'assessing', 'reviewing', 'scenario', 'thesis', 'verdict', 'conclusion']):
                is_header = True
                clean_title = line.title()
            elif i == 0 and len(line) < 50:
                is_header = True
                clean_title = line.title()
                
        if is_header:
            if in_section:
                html_out += '</div></details>'
            
            icon = "🔹"
            llc = line.lower()
            if "verdict" in llc or "conclusion" in llc: icon = "⚖️"
            elif "data" in llc or "metric" in llc or "ratio" in llc: icon = "📊"
            elif "action" in llc or "step" in llc or "phase" in llc: icon = "🎯"
            elif "risk" in llc: icon = "⚠️"
            elif "thesis" in llc: icon = "💡"
            elif "target" in llc or "stop" in llc: icon = "📌"
            elif "calculat" in llc: icon = "🧮"
            elif "analyz" in llc or "evaluat" in llc or "review" in llc: icon = "🔍"
            elif "financial" in llc or "balance" in llc or "cash" in llc: icon = "💰"
            if i == 0 and icon == "🔹": icon = "🧠"
            
            html_out += f'''<details open style="margin-bottom: 10px;">
<summary style="cursor: pointer; outline: none; margin-bottom: 4px; color: #a5b4fc; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.05em;">
<span style="text-transform: uppercase;">{icon} {clean_title}</span>
</summary>
<div style="margin-left: 12px; border-left: 2px solid #334155; padding-left: 16px; margin-top: 4px; margin-bottom: 8px; color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;">'''
            in_section = True
        else:
            if not in_section:
                html_out += f'''<details open style="margin-bottom: 10px;">
<summary style="cursor: pointer; outline: none; margin-bottom: 4px; color: #a5b4fc; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.05em;">
<span style="text-transform: uppercase;">🧠 Thinking Process</span>
</summary>
<div style="margin-left: 12px; border-left: 2px solid #334155; padding-left: 16px; margin-top: 4px; margin-bottom: 8px; color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;">'''
                in_section = True
                
            if line.startswith('-') or line.startswith('*'):
                line_content = line[1:].strip()
                html_out += f'<div style="margin-bottom: 4px;"><span style="color: #64748b; margin-right: 6px;">•</span> {line_content}</div>'
            else:
                html_out += f'<div style="margin-bottom: 4px;">{line}</div>'
            
    if in_section:
        html_out += '</div></details>'
        
    html_out += '</div>'
    return html_out
