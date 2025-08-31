import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
from io import BytesIO
import numpy as np
from typing import List, Tuple, Dict

st.set_page_config(page_title="FMCG Duplicate Detection Tool", layout="wide")

st.title("🏭 FMCG Duplicate Detection Tool")
st.write("Advanced Material Description Analysis & Duplicate Management")

# --- Helper functions ---
def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing for better matching"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    
    # Remove common FMCG-specific noise words
    noise_words = {
        'pack', 'pkt', 'packet', 'bottle', 'jar', 'can', 'box', 'tube', 'sachet',
        'ml', 'gm', 'kg', 'ltr', 'litre', 'gram', 'kilogram', 'milliliter',
        'pc', 'pcs', 'piece', 'pieces', 'unit', 'units', 'each'
    }
    
    # Standardize units and measurements
    text = re.sub(r'\b(\d+)\s*(ml|gm|kg|ltr|g|l)\b', r'\1\2', text)
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove noise words
    words = text.split()
    words = [word for word in words if word not in noise_words]
    
    # Remove extra spaces
    return ' '.join(words)

def extract_features(text: str) -> Dict:
    """Extract key features from product description"""
    if pd.isna(text):
        return {"brand": "", "size": "", "variant": "", "core_product": ""}
    
    text = str(text)
    
    # Extract size/quantity patterns
    size_pattern = r'(\d+(?:\.\d+)?)\s*(ml|gm|kg|ltr|g|l|oz|lb)'
    size_match = re.search(size_pattern, text.lower())
    size = size_match.group() if size_match else ""
    
    # Extract brand (assuming first word or words in caps)
    brand_pattern = r'^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)'
    brand_match = re.search(brand_pattern, text.strip())
    brand = brand_match.group(1) if brand_match else ""
    
    # Extract variant indicators
    variant_keywords = ['lite', 'light', 'diet', 'zero', 'max', 'ultra', 'premium', 'organic', 'natural']
    variant = ""
    for keyword in variant_keywords:
        if keyword in text.lower():
            variant += f" {keyword}"
    
    # Core product (remove brand, size, variant)
    core = text
    if brand:
        core = core.replace(brand, "")
    if size:
        core = core.replace(size_match.group(), "")
    core = preprocess_text(core)
    
    return {
        "brand": brand.lower(),
        "size": size.lower(),
        "variant": variant.strip().lower(),
        "core_product": core
    }

def calculate_similarity_score(desc1: str, desc2: str) -> Tuple[float, str]:
    """Calculate comprehensive similarity score between two descriptions"""
    if pd.isna(desc1) or pd.isna(desc2):
        return 0.0, "invalid"
    
    # Extract features
    features1 = extract_features(desc1)
    features2 = extract_features(desc2)
    
    # Exact match check
    if preprocess_text(desc1) == preprocess_text(desc2):
        return 1.0, "exact_match"
    
    # Feature-based scoring
    scores = {}
    
    # Brand similarity (high weight)
    if features1["brand"] and features2["brand"]:
        brand_sim = SequenceMatcher(None, features1["brand"], features2["brand"]).ratio()
        scores["brand"] = brand_sim * 0.3
    else:
        scores["brand"] = 0
    
    # Size similarity (medium weight)
    if features1["size"] and features2["size"]:
        size_sim = 1.0 if features1["size"] == features2["size"] else 0.0
        scores["size"] = size_sim * 0.2
    else:
        scores["size"] = 0
    
    # Core product similarity (high weight)
    core_sim = SequenceMatcher(None, features1["core_product"], features2["core_product"]).ratio()
    scores["core"] = core_sim * 0.4
    
    # Overall text similarity (medium weight)
    text_sim = SequenceMatcher(None, preprocess_text(desc1), preprocess_text(desc2)).ratio()
    scores["text"] = text_sim * 0.1
    
    total_score = sum(scores.values())
    
    # Determine duplicate type
    if total_score >= 0.95:
        dup_type = "exact_duplicate"
    elif total_score >= 0.8:
        dup_type = "high_similarity"
    elif total_score >= 0.6:
        dup_type = "medium_similarity"
    elif total_score >= 0.4:
        dup_type = "low_similarity"
    else:
        dup_type = "unique"
    
    return total_score, dup_type

def get_duplicate_color(status: str, score: float) -> str:
    """Get color based on duplicate status and score"""
    if status == "unique":
        return "#f0f0f0"  # Light gray
    elif score >= 0.95:
        return "#ffcdd2"  # Light red - exact duplicates
    elif score >= 0.8:
        return "#ffe0b2"  # Light orange - high similarity
    elif score >= 0.6:
        return "#fff9c4"  # Light yellow - medium similarity
    else:
        return "#e8f5e8"  # Light green - low similarity

def process_duplicates(df: pd.DataFrame, desc_col: str, similarity_threshold: float) -> pd.DataFrame:
    """Process duplicates with enhanced logic"""
    processed_rows = []
    n = len(df)
    
    # Initialize progress tracking
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n):
        progress_bar.progress((i + 1) / n)
        status_text.text(f"Processing record {i + 1} of {n}")
        
        row = df.iloc[i]
        desc = row[desc_col]
        
        max_similarity = 0.0
        best_match_idx = -1
        
        # Check against all other rows
        for j in range(n):
            if i != j:
                other_desc = df.iloc[j][desc_col]
                similarity, dup_type = calculate_similarity_score(desc, other_desc)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_idx = j
        
        # Determine duplicate status
        if max_similarity >= similarity_threshold:
            duplicate_status = "duplicate"
            confidence = max_similarity
        else:
            duplicate_status = "unique"
            confidence = 1.0 - max_similarity
        
        processed_rows.append({
            **row.to_dict(),
            "similarity_score": round(max_similarity, 3),
            "duplicate_status": duplicate_status,
            "confidence": round(confidence, 3),
            "best_match_index": best_match_idx if best_match_idx != -1 else None,
            "keep_record": duplicate_status == "unique"  # Default: keep unique records
        })
    
    progress_bar.empty()
    status_text.empty()
    st.session_state.processing_complete = True
    
    return pd.DataFrame(processed_rows)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.8, 
    step=0.05,
    help="Higher values = stricter duplicate detection"
)

# --- File upload ---
uploaded_file = st.file_uploader(
    "📊 Upload Excel/CSV File", 
    type=["xlsx", "xls", "csv"],
    help="Upload your FMCG inventory file with material descriptions"
)

if uploaded_file:
    try:
        # Load data
        with st.spinner("📂 Loading file..."):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        st.success(f"✅ File loaded successfully! ({len(df)} records found)")

        # Auto-detect material description column
        desc_col = None
        possible_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            if "material" in col_lower and "description" in col_lower:
                desc_col = col
                break
            elif any(keyword in col_lower for keyword in ["description", "material", "product", "item"]):
                possible_columns.append(col)
        
        if not desc_col and possible_columns:
            desc_col = possible_columns[0]
        
        # Column selection
        if desc_col:
            desc_col = st.sidebar.selectbox(
                "Material Description Column:", 
                df.columns, 
                index=list(df.columns).index(desc_col)
            )
        else:
            desc_col = st.selectbox("Select Material Description Column:", df.columns)

        if desc_col and len(df) > 0:
            # Show sample data
            st.subheader("📋 Sample Data Preview")
            st.dataframe(df.head(10))
            
            # Process button
            if st.button("🔍 Analyze Duplicates"):
                with st.spinner("🔍 Analyzing duplicates... This may take a moment."):
                    df_processed = process_duplicates(df, desc_col, similarity_threshold)
                    st.session_state.df_processed = df_processed
            
            # Show results if processing is complete
            if 'df_processed' in st.session_state:
                df_processed = st.session_state.df_processed
                
                # --- Enhanced Statistics ---
                st.subheader("📊 Analysis Results")
                
                total_records = len(df_processed)
                unique_records = len(df_processed[df_processed["duplicate_status"] == "unique"])
                duplicate_records = total_records - unique_records
                selected_records = len(df_processed[df_processed["keep_record"] == True])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", total_records)
                col2.metric("Unique Records", unique_records, f"{unique_records/total_records*100:.1f}%")
                col3.metric("Duplicate Records", duplicate_records, f"{duplicate_records/total_records*100:.1f}%")
                col4.metric("Selected to Keep", selected_records, f"{selected_records/total_records*100:.1f}%")
                
                # Similarity distribution chart
                st.subheader("📈 Data Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Similarity Score Distribution**")
                    similarity_scores = df_processed["similarity_score"]
                    hist_data = pd.cut(similarity_scores, 
                                     bins=[0, 0.4, 0.6, 0.8, 0.95, 1.0], 
                                     labels=["Low (0-0.4)", "Medium (0.4-0.6)", "High (0.6-0.8)", "Very High (0.8-0.95)", "Exact (0.95-1.0)"])
                    dist_counts = hist_data.value_counts().sort_index()
                    st.bar_chart(dist_counts)
                
                with col2:
                    st.write("**Duplicate Status Distribution**")
                    status_counts = df_processed["duplicate_status"].value_counts()
                    
                    # Create pie chart visualization
                    pie_chart_data = []
                    colors = []
                    for status, count in status_counts.items():
                        percentage = count / len(df_processed) * 100
                        pie_chart_data.append(f"{status.title()}: {count} ({percentage:.1f}%)")
                        colors.append("#ff6b6b" if status == "duplicate" else "#51cf66")
                    
                    # Simple text-based pie chart representation
                    st.write("**📊 Status Breakdown:**")
                    for i, (status, count) in enumerate(status_counts.items()):
                        percentage = count / len(df_processed) * 100
                        # Create a simple bar representation
                        bar_length = int(percentage / 2)  # Scale down for display
                        bar = "█" * bar_length + "░" * (50 - bar_length)
                        color = "🔴" if status == "duplicate" else "🟢"
                        st.write(f"{color} **{status.title()}**: {count} records ({percentage:.1f}%)")
                        st.write(f"   {bar}")
                    
                    # Also show the raw chart data
                    chart_data = pd.DataFrame({
                        'Status': status_counts.index,
                        'Count': status_counts.values
                    })
                    st.bar_chart(chart_data.set_index('Status'))
                
                # --- Filters ---
                st.subheader("🎯 Advanced Filter Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Status & Similarity Filters**")
                    status_filter = st.multiselect(
                        "Filter by Status:",
                        ["unique", "duplicate"],
                        default=["unique", "duplicate"]
                    )
                    
                    similarity_range = st.slider(
                        "Similarity Score Range:",
                        0.0, 1.0, (0.0, 1.0), 0.05
                    )
                
                with col2:
                    st.write("**Content Filters**")
                    show_only_selected = st.checkbox("Show only records marked to keep", False)
                    show_features = st.checkbox("Show extracted features", False)
                
                # --- Text/Word Search Filters ---
                st.write("**🔍 Search & Select by Words/Letters**")
                
                search_col1, search_col2, search_col3 = st.columns(3)
                
                with search_col1:
                    search_text = st.text_input(
                        "Search in descriptions:",
                        placeholder="e.g. CHILLI, SPI-, (ME), etc.",
                        help="Enter words or letters to search for in material descriptions"
                    )
                    
                    search_mode = st.selectbox(
                        "Search Mode:",
                        ["Contains", "Starts with", "Ends with", "Exact match"],
                        help="How to match your search text"
                    )
                
                with search_col2:
                    if search_text:
                        # Count matches
                        if search_mode == "Contains":
                            matches = df_processed[df_processed[desc_col].str.contains(search_text, case=False, na=False)]
                        elif search_mode == "Starts with":
                            matches = df_processed[df_processed[desc_col].str.startswith(search_text, na=False)]
                        elif search_mode == "Ends with":
                            matches = df_processed[df_processed[desc_col].str.endswith(search_text, na=False)]
                        else:  # Exact match
                            matches = df_processed[df_processed[desc_col].str.lower() == search_text.lower()]
                        
                        st.metric("Search Results", len(matches))
                        
                        if len(matches) > 0:
                            if st.button("✅ Select All Search Results"):
                                # Select all matching records
                                mask = df_processed[desc_col].isin(matches[desc_col])
                                df_processed.loc[mask, "keep_record"] = True
                                st.session_state.df_processed = df_processed
                                st.success(f"Selected {len(matches)} matching records")
                                st.experimental_rerun()
                            
                            if st.button("❌ Deselect All Search Results"):
                                # Deselect all matching records
                                mask = df_processed[desc_col].isin(matches[desc_col])
                                df_processed.loc[mask, "keep_record"] = False
                                st.session_state.df_processed = df_processed
                                st.success(f"Deselected {len(matches)} matching records")
                                st.experimental_rerun()
                
                with search_col3:
                    # Quick filter buttons for common FMCG patterns
                    st.write("**Quick Filters:**")
                    if st.button("🌶️ Select All CHILLI"):
                        mask = df_processed[desc_col].str.contains("CHILLI", case=False, na=False)
                        df_processed.loc[mask, "keep_record"] = True
                        st.session_state.df_processed = df_processed
                        st.success("Selected all CHILLI items")
                        st.experimental_rerun()
                    
                    if st.button("🧄 Select All GARLIC"):
                        mask = df_processed[desc_col].str.contains("GARLIC", case=False, na=False)
                        df_processed.loc[mask, "keep_record"] = True
                        st.session_state.df_processed = df_processed
                        st.success("Selected all GARLIC items")
                        st.experimental_rerun()
                    
                    if st.button("🔤 Select All SPI- Codes"):
                        mask = df_processed[desc_col].str.contains("SPI-", case=False, na=False)
                        df_processed.loc[mask, "keep_record"] = True
                        st.session_state.df_processed = df_processed
                        st.success("Selected all SPI- codes")
                        st.experimental_rerun()
                    
                    if st.button("🌍 Select All (EU-23)"):
                        mask = df_processed[desc_col].str.contains("(EU-23)", case=False, na=False)
                        df_processed.loc[mask, "keep_record"] = True
                        st.session_state.df_processed = df_processed
                        st.success("Selected all EU-23 items")
                        st.experimental_rerun()
                
                # --- Pattern-based Selection ---
                st.write("**🎯 Advanced Pattern Selection**")
                
                pattern_col1, pattern_col2 = st.columns(2)
                
                with pattern_col1:
                    region_pattern = st.selectbox(
                        "Select by Region/Origin:",
                        ["All", "(ME)", "(NA)", "(CHI)", "(LOC)", "(EU-23)", "(ID)", "(MRL)"],
                        help="Select records from specific regions"
                    )
                    
                    if region_pattern != "All" and st.button(f"Select All {region_pattern} Items"):
                        mask = df_processed[desc_col].str.contains(region_pattern, case=False, na=False)
                        df_processed.loc[mask, "keep_record"] = True
                        st.session_state.df_processed = df_processed
                        count = mask.sum()
                        st.success(f"Selected {count} items from {region_pattern}")
                        st.experimental_rerun()
                
                with pattern_col2:
                    product_category = st.selectbox(
                        "Select by Product Category:",
                        ["All", "DEHYDRATED", "FRESH", "SEED", "PDR", "OIL", "SALT", "SUGAR", "FLV-", "THK-", "CHILLI"],
                        help="Select records by product type"
                    )
                    
                    if product_category != "All" and st.button(f"Select All {product_category} Items"):
                        mask = df_processed[desc_col].str.contains(product_category, case=False, na=False)
                        df_processed.loc[mask, "keep_record"] = True
                        st.session_state.df_processed = df_processed
                        count = mask.sum()
                        st.success(f"Selected {count} {product_category} items")
                        st.experimental_rerun()
                
                # Apply all filters
                filtered_df = df_processed[
                    (df_processed["duplicate_status"].isin(status_filter)) &
                    (df_processed["similarity_score"] >= similarity_range[0]) &
                    (df_processed["similarity_score"] <= similarity_range[1])
                ]
                
                # Apply search filter if search text is provided
                if search_text:
                    if search_mode == "Contains":
                        search_mask = filtered_df[desc_col].str.contains(search_text, case=False, na=False)
                    elif search_mode == "Starts with":
                        search_mask = filtered_df[desc_col].str.startswith(search_text, na=False)
                    elif search_mode == "Ends with":
                        search_mask = filtered_df[desc_col].str.endswith(search_text, na=False)
                    else:  # Exact match
                        search_mask = filtered_df[desc_col].str.lower() == search_text.lower()
                    
                    filtered_df = filtered_df[search_mask]
                    
                    if len(filtered_df) > 0:
                        st.info(f"🔍 Showing {len(filtered_df)} records matching '{search_text}'")
                    else:
                        st.warning(f"🔍 No records found matching '{search_text}'")
                
                if show_only_selected:
                    filtered_df = filtered_df[filtered_df["keep_record"] == True]
                
                # --- Bulk Actions ---
                st.subheader("⚡ Bulk Selection Actions")
                
                st.write("**General Actions:**")
                bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
                
                with bulk_col1:
                    if st.button("✅ Keep Only Unique Records"):
                        df_processed["keep_record"] = df_processed["duplicate_status"] == "unique"
                        st.session_state.df_processed = df_processed
                        st.success("Selected all unique records")
                        st.experimental_rerun()
                
                with bulk_col2:
                    if st.button("🔄 Keep All Records"):
                        df_processed["keep_record"] = True
                        st.session_state.df_processed = df_processed
                        st.success("Selected all records")
                        st.experimental_rerun()
                
                with bulk_col3:
                    if st.button("❌ Deselect All Duplicates"):
                        df_processed["keep_record"] = df_processed["duplicate_status"] == "unique"
                        st.session_state.df_processed = df_processed
                        st.success("Deselected all duplicate records")
                        st.experimental_rerun()
                
                # Show current selection stats
                selected_count = len(df_processed[df_processed["keep_record"] == True])
                total_count = len(df_processed)
                reduction = (1 - selected_count/total_count) * 100
                st.write(f"**Current Selection**: {selected_count}/{total_count} records ({reduction:.1f}% reduction)")
                
                # --- Data Preview ---
                st.subheader("📋 Data Analysis")
                
                # Prepare display columns
                display_cols = [desc_col, "similarity_score", "duplicate_status", "confidence", "keep_record"]
                
                if show_features:
                    # Add feature columns
                    feature_data = []
                    for idx, row in filtered_df.iterrows():
                        features = extract_features(row[desc_col])
                        feature_data.append({
                            "Index": idx,
                            "Brand": features["brand"],
                            "Size": features["size"],
                            "Core Product": features["core_product"]
                        })
                    
                    if feature_data:
                        feature_df = pd.DataFrame(feature_data)
                        st.subheader("🔍 Extracted Features")
                        st.dataframe(feature_df, use_container_width=True)
                
                # Show main data with color coding
                def style_dataframe(df_to_style):
                    def highlight_rows(row):
                        if row["duplicate_status"] == "unique":
                            color = "#f0f0f0"  # Light gray
                        elif row["similarity_score"] >= 0.95:
                            color = "#ffcdd2"  # Light red
                        elif row["similarity_score"] >= 0.8:
                            color = "#ffe0b2"  # Light orange
                        elif row["similarity_score"] >= 0.6:
                            color = "#fff9c4"  # Light yellow
                        else:
                            color = "#e8f5e8"  # Light green
                        
                        return [f"background-color: {color}" for _ in row]
                    
                    return df_to_style.style.apply(highlight_rows, axis=1)
                
                styled_df = style_dataframe(filtered_df[display_cols])
                st.dataframe(styled_df)
                
                # Color legend
                st.markdown("""
                **🎨 Color Legend:**
                - 🔴 **Red**: Exact duplicates (95-100% similarity)
                - 🟠 **Orange**: High similarity (80-95%)
                - 🟡 **Yellow**: Medium similarity (60-80%)
                - 🟢 **Green**: Low similarity (40-60%)
                - ⚪ **Gray**: Unique records
                """)
                
                # --- Duplicate Groups Analysis ---
                duplicates_only = df_processed[df_processed["duplicate_status"] == "duplicate"]
                if len(duplicates_only) > 0:
                    st.subheader("🔍 Duplicate Analysis Details")
                    
                    # Group duplicates by similarity
                    high_sim = duplicates_only[duplicates_only["similarity_score"] >= 0.8]
                    medium_sim = duplicates_only[(duplicates_only["similarity_score"] >= 0.6) & 
                                               (duplicates_only["similarity_score"] < 0.8)]
                    
                    if len(high_sim) > 0:
                        st.write(f"**High Similarity Duplicates:** {len(high_sim)} records")
                        with st.expander("View High Similarity Records"):
                            st.dataframe(high_sim[[desc_col, "similarity_score", "keep_record"]])
                    
                    if len(medium_sim) > 0:
                        st.write(f"**Medium Similarity Duplicates:** {len(medium_sim)} records")
                        with st.expander("View Medium Similarity Records"):
                            st.dataframe(medium_sim[[desc_col, "similarity_score", "keep_record"]])
                
                # --- Export Options ---
                st.subheader("📥 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    export_option = st.selectbox(
                        "What to export:",
                        ["Selected Records Only", "All Records with Analysis", "Unique Records Only"]
                    )
                
                with col2:
                    include_analysis = st.checkbox("Include analysis columns", True)
                
                if st.button("📊 Generate Export File"):
                    try:
                        # Prepare export data
                        if export_option == "Selected Records Only":
                            export_df = df_processed[df_processed["keep_record"] == True].copy()
                        elif export_option == "Unique Records Only":
                            export_df = df_processed[df_processed["duplicate_status"] == "unique"].copy()
                        else:
                            export_df = df_processed.copy()
                        
                        # Remove analysis columns if not needed
                        if not include_analysis:
                            analysis_cols = ["similarity_score", "duplicate_status", "confidence", 
                                           "best_match_index", "keep_record"]
                            export_df = export_df.drop(columns=[col for col in analysis_cols 
                                                              if col in export_df.columns])
                        
                        # Prepare export data
                        if export_option == "Selected Records Only":
                            export_df = df_processed[df_processed["keep_record"] == True].copy()
                        elif export_option == "Unique Records Only":
                            export_df = df_processed[df_processed["duplicate_status"] == "unique"].copy()
                        else:
                            export_df = df_processed.copy()
                        
                        # Remove analysis columns if not needed
                        if not include_analysis:
                            analysis_cols = ["similarity_score", "duplicate_status", "confidence", 
                                           "best_match_index", "keep_record"]
                            export_df = export_df.drop(columns=[col for col in analysis_cols 
                                                              if col in export_df.columns])
                        
                        # Clean data for Excel export - handle NaN and infinite values
                        export_df = export_df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
                        export_df = export_df.fillna('')  # Replace NaN with empty string
                        
                        # Ensure numeric columns are properly formatted
                        numeric_cols = export_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            export_df[col] = pd.to_numeric(export_df[col], errors='coerce').fillna(0)
                        
                        # Generate Excel file with enhanced formatting
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            export_df.to_excel(writer, sheet_name="FMCG_Cleaned_Data", index=False)
                            
                            workbook = writer.book
                            worksheet = writer.sheets["FMCG_Cleaned_Data"]
                            
                            # Define color formats matching the view
                            color_formats = {
                                "unique": workbook.add_format({"bg_color": "#f0f0f0", "border": 1}),  # Light gray
                                "exact": workbook.add_format({"bg_color": "#ffcdd2", "border": 1}),   # Light red
                                "high": workbook.add_format({"bg_color": "#ffe0b2", "border": 1}),    # Light orange
                                "medium": workbook.add_format({"bg_color": "#fff9c4", "border": 1}),  # Light yellow
                                "low": workbook.add_format({"bg_color": "#e8f5e8", "border": 1}),     # Light green
                                "default": workbook.add_format({"border": 1})
                            }
                            
                            # Apply row coloring based on similarity scores
                            for row_idx in range(len(export_df)):
                                try:
                                    if "duplicate_status" in export_df.columns and "similarity_score" in export_df.columns:
                                        status = export_df.iloc[row_idx]["duplicate_status"]
                                        score = export_df.iloc[row_idx]["similarity_score"]
                                        
                                        # Determine format based on status and score
                                        if status == "unique":
                                            format_to_use = color_formats["unique"]
                                        elif score >= 0.95:
                                            format_to_use = color_formats["exact"]
                                        elif score >= 0.8:
                                            format_to_use = color_formats["high"]
                                        elif score >= 0.6:
                                            format_to_use = color_formats["medium"]
                                        elif score >= 0.4:
                                            format_to_use = color_formats["low"]
                                        else:
                                            format_to_use = color_formats["default"]
                                    else:
                                        format_to_use = color_formats["default"]
                                    
                                    # Apply format to entire row
                                    for col_idx in range(len(export_df.columns)):
                                        cell_value = export_df.iloc[row_idx, col_idx]
                                        # Convert to string to avoid NaN issues
                                        if pd.isna(cell_value):
                                            cell_value = ""
                                        worksheet.write(row_idx + 1, col_idx, str(cell_value), format_to_use)
                                        
                                except Exception as row_error:
                                    # If individual row fails, continue with default formatting
                                    for col_idx in range(len(export_df.columns)):
                                        cell_value = str(export_df.iloc[row_idx, col_idx]) if not pd.isna(export_df.iloc[row_idx, col_idx]) else ""
                                        worksheet.write(row_idx + 1, col_idx, cell_value, color_formats["default"])
                            
                            # Auto-adjust column widths
                            for col_idx, col in enumerate(export_df.columns):
                                try:
                                    max_length = max(
                                        export_df[col].astype(str).map(len).max(),
                                        len(str(col))
                                    )
                                    worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
                                except:
                                    worksheet.set_column(col_idx, col_idx, 15)  # Default width
                            
                            # Add a legend sheet
                            legend_sheet = workbook.add_worksheet("Color_Legend")
                            
                            # Header
                            header_format = workbook.add_format({"bold": True, "font_size": 14, "bg_color": "#e0e0e0"})
                            legend_sheet.write(0, 0, "FMCG Duplicate Detection - Color Legend", header_format)
                            
                            # Legend entries
                            legend_data = [
                                ("Status/Similarity", "Color", "Description"),
                                ("Unique Records", "Light Gray", "No similar records found"),
                                ("Exact Duplicates (95-100%)", "Light Red", "Nearly identical descriptions"),
                                ("High Similarity (80-95%)", "Light Orange", "Very similar, likely duplicates"),
                                ("Medium Similarity (60-80%)", "Light Yellow", "Moderately similar, review needed"),
                                ("Low Similarity (40-60%)", "Light Green", "Some similarity, probably different"),
                            ]
                            
                            for i, (status, color, desc) in enumerate(legend_data):
                                if i == 0:  # Header row
                                    format_obj = workbook.add_format({"bold": True, "border": 1})
                                elif "Unique" in status:
                                    format_obj = color_formats["unique"]
                                elif "Exact" in status:
                                    format_obj = color_formats["exact"]
                                elif "High" in status:
                                    format_obj = color_formats["high"]
                                elif "Medium" in status:
                                    format_obj = color_formats["medium"]
                                elif "Low" in status:
                                    format_obj = color_formats["low"]
                                else:
                                    format_obj = workbook.add_format({"border": 1})
                                
                                legend_sheet.write(i + 2, 0, status, format_obj)
                                legend_sheet.write(i + 2, 1, color, format_obj)
                                legend_sheet.write(i + 2, 2, desc, format_obj)
                            
                            # Set column widths for legend
                            legend_sheet.set_column(0, 0, 25)
                            legend_sheet.set_column(1, 1, 15)
                            legend_sheet.set_column(2, 2, 35)
                        
                        excel_data = output.getvalue()
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Cleaned Data",
                            data=excel_data,
                            file_name=f"FMCG_Cleaned_{export_option.replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success(f"✅ Export ready! {len(export_df)} records prepared for download.")
                        
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
                
                # --- Summary Report ---
                st.subheader("📋 Processing Summary")
                
                reduction_percentage = (1 - selected_records/total_records) * 100
                avg_similarity = df_processed["similarity_score"].mean()
                
                summary_metrics = {
                    "📊 **Total Records Processed**": f"{total_records:,}",
                    "✅ **Unique Records Found**": f"{unique_records:,} ({unique_records/total_records*100:.1f}%)",
                    "🔄 **Duplicate Records**": f"{duplicate_records:,} ({duplicate_records/total_records*100:.1f}%)",
                    "📦 **Records Selected to Keep**": f"{selected_records:,}",
                    "📉 **Potential Data Reduction**": f"{reduction_percentage:.1f}%",
                    "📈 **Average Similarity Score**": f"{avg_similarity:.3f}"
                }
                
                for metric, value in summary_metrics.items():
                    st.write(f"{metric}: {value}")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("**Troubleshooting tips:**")
        st.write("- Ensure your file has a column with material descriptions")
        st.write("- Check that the file is not corrupted")
        st.write("- Try uploading a smaller sample first")

else:
    # --- Help Section ---
    st.subheader("📖 How to Use This Enhanced Tool")
    
    st.markdown("""
    ### 🎯 **What This Tool Does**
    This advanced FMCG duplicate detection tool helps you:
    - **Identify duplicate** material descriptions in your inventory
    - **Clean your data** by removing or consolidating duplicates
    - **Analyze similarity patterns** in your product database
    - **Export clean datasets** for further use
    
    ### 📋 **Step-by-Step Instructions**
    1. **📊 Upload** your Excel or CSV file containing material descriptions
    2. **⚙️ Configure** the similarity threshold (0.8 recommended for most cases)
    3. **🔍 Analyze** - Click the "Analyze Duplicates" button
    4. **🎯 Filter** results using the available options
    5. **⚡ Use bulk actions** to quickly select records to keep
    6. **📥 Export** your cleaned data
    
    ### 🧠 **Smart Detection Features**
    - **Brand Recognition**: Identifies and compares product brands
    - **Size Standardization**: Normalizes different unit formats (ml, gm, etc.)
    - **Noise Filtering**: Removes common packaging terms
    - **Feature Extraction**: Analyzes core product characteristics
    - **Weighted Scoring**: Prioritizes important attributes
    
    ### 📊 **Understanding Similarity Scores**
    - **0.95-1.0**: Exact or near-exact duplicates
    - **0.8-0.95**: High similarity (likely duplicates)
    - **0.6-0.8**: Medium similarity (review recommended)
    - **0.4-0.6**: Low similarity (probably different products)
    - **0.0-0.4**: Unique products
    
    ### 💡 **Best Practices**
    - Start with a **similarity threshold of 0.8**
    - **Review high-similarity groups** manually
    - **Test with a sample** before processing large datasets
    - **Keep backups** of your original data
    """)
    
    st.info("💡 **Pro Tip**: Upload a small sample file first to test the similarity threshold that works best for your data!")
    
    # Sample data format
    st.subheader("📝 Expected File Format")
    sample_data = pd.DataFrame({
        "Material Code": ["MAT001", "MAT002", "MAT003"],
        "Material Description": [
            "Coca Cola 330ml Bottle Pack",
            "Coca-Cola 330 ml Bottle Package", 
            "Pepsi 500ml Plastic Bottle"
        ],
        "Category": ["Beverages", "Beverages", "Beverages"],
        "Price": [25.50, 25.50, 30.00]
    })
    st.dataframe(sample_data)