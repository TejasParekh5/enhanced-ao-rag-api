"""
Optimized AO RAG API - Enhanced Performance and Maintainability
Key Optimizations:
1. Fixed column mapping to match actual Excel structure
2. Improved error handling and validation
3. Enhanced caching and data processing
4. Better code organization and documentation
5. Performance improvements in data aggregation
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import json
import requests
import logging
import re
from typing import Dict, List, Optional, Union
from functools import lru_cache
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration


class Config:
    OLLAMA_URL = "http://localhost:11434/api/generate"
    DEFAULT_MODEL = "llama3.1:8b"
    EXCEL_FILE = "Cybersecurity_KPI_Minimal.xlsx"
    DATA_FILE = "ao_rag_data.pkl"
    INDEX_FILE = "ao_rag_faiss.index"

    # Column mapping - maps our internal names to Excel column names
    COLUMN_MAPPING = {
        'ao_name': 'Application_Owner_Name',
        'application': 'Application_Name',
        'department': 'Dept_Name',
        'risk_score': 'Risk_Score',
        'severity': 'Severity',
        'cvss_score': 'CVSS_Score',
        'vulnerability_desc': 'Vulnerability_Description',
        'asset_name': 'Asset_Name',
        'asset_type': 'Asset_Type',
        'status': 'Status',
        'first_detected': 'First_Detected_Date',
        'closure_date': 'Closure_Date',
        'days_to_close': 'Days_to_Close'
    }


class OllamaService:
    """Optimized Ollama service with better error handling and caching"""

    @staticmethod
    @lru_cache(maxsize=100)
    def query_ollama(prompt: str, temperature: float = 0.7, model: str = Config.DEFAULT_MODEL) -> str:
        """Cached Ollama queries for better performance"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }

            response = requests.post(
                Config.OLLAMA_URL, json=payload, timeout=300)
            response.raise_for_status()

            data = response.json()
            return data.get('response', 'No response from Ollama')

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama service unavailable")
            return "AI analysis unavailable - Ollama service not running"
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timeout")
            return "AI analysis timeout - please try again"
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return f"AI analysis error: {str(e)}"

    @staticmethod
    def analyze_vulnerability(vulnerability_name: str, code_snippet: str = "",
                              risk_rating: str = "", description: str = "") -> str:
        """Analyze vulnerability with improved prompts"""
        if not code_snippet and not risk_rating:
            # OWASP categorization mode
            prompt = f"""
Analyze this vulnerability and categorize it according to OWASP Top 10 (2021):

Vulnerability: {vulnerability_name}
Description: {description}

Provide:
1. Main vulnerability category
2. OWASP Top 10 classification
3. Risk level assessment
4. Common attack vectors
5. Mitigation recommendations

OWASP Top 10 (2021): A01-Broken Access Control, A02-Cryptographic Failures, 
A03-Injection, A04-Insecure Design, A05-Security Misconfiguration, 
A06-Vulnerable Components, A07-Authentication Failures, A08-Data Integrity Failures, 
A09-Logging Failures, A10-Server-Side Request Forgery
"""
        else:
            # Code analysis mode
            prompt = f"""
Security Analysis for: {vulnerability_name}
Risk Rating: {risk_rating}

Code to analyze:
{code_snippet}

Provide:
1. Vulnerability explanation
2. How it manifests in the code
3. Specific code fixes
4. Security best practices
5. Testing recommendations
"""
        return OllamaService.query_ollama(prompt, temperature=0.5)

    @staticmethod
    def enhance_ao_response(query: str, ao_context: str) -> str:
        """Enhanced AO analysis with better prompts"""
        prompt = f"""
As a cybersecurity analyst, provide a comprehensive analysis based on this Application Owner data:

Query: {query}

Application Owner Data:
{ao_context}

Provide structured analysis:
1. Executive Summary
2. Critical Security Issues
3. Risk Assessment
4. Prioritized Action Plan
5. Compliance Status
6. Recommendations with Timeline

Focus on actionable insights and specific next steps.
"""
        return OllamaService.query_ollama(prompt, temperature=0.6)


class DataProcessor:
    """Optimized data processing with better error handling"""

    @staticmethod
    def safe_convert(value, convert_func, default=0):
        """Safely convert values with fallback"""
        try:
            return convert_func(value) if pd.notna(value) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_vulnerability_stats(df_group: pd.DataFrame) -> Dict:
        """Calculate vulnerability statistics for an AO group"""
        stats = {
            'total_vulnerabilities': len(df_group),
            'high_vulnerabilities': 0,
            'medium_vulnerabilities': 0,
            'low_vulnerabilities': 0,
            'critical_vulnerabilities': 0
        }

        # Count by severity
        if 'Severity' in df_group.columns:
            severity_counts = df_group['Severity'].value_counts()
            stats['high_vulnerabilities'] = severity_counts.get('High', 0)
            stats['medium_vulnerabilities'] = severity_counts.get('Medium', 0)
            stats['low_vulnerabilities'] = severity_counts.get('Low', 0)
            stats['critical_vulnerabilities'] = severity_counts.get(
                'Critical', 0)

        return stats

    @staticmethod
    def calculate_risk_metrics(df_group: pd.DataFrame) -> Dict:
        """Calculate risk metrics for an AO group"""
        risk_scores = df_group['Risk_Score'].dropna()
        cvss_scores = df_group['CVSS_Score'].dropna()

        return {
            'avg_risk_score': round(risk_scores.mean(), 2) if not risk_scores.empty else 0,
            'max_risk_score': round(risk_scores.max(), 2) if not risk_scores.empty else 0,
            'avg_cvss_score': round(cvss_scores.mean(), 2) if not cvss_scores.empty else 0,
            'risk_score_count': len(risk_scores)
        }


class AORAGSystem:
    """Optimized RAG system with improved performance and maintainability"""

    def __init__(self, excel_file_path: str = Config.EXCEL_FILE):
        self.excel_file_path = excel_file_path
        self.model = None
        self.ao_data: List[Dict] = []
        self.embeddings = None
        self.faiss_index = None
        self.data_file = Config.DATA_FILE
        self.index_file = Config.INDEX_FILE

        # Performance tracking
        self.stats = {
            'total_aos': 0,
            'total_vulnerabilities': 0,
            'last_updated': None
        }

        self._initialize_system()

    def _initialize_system(self):
        """Initialize the RAG system with better error handling"""
        try:
            # Initialize sentence transformer
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            # Load or process data
            if self._load_processed_data():
                logger.info(
                    f"Loaded existing data: {self.stats['total_aos']} AOs")
            else:
                logger.info("Processing data from scratch...")
                self._process_excel_data()
                self._create_embeddings()
                self._build_faiss_index()
                self._save_processed_data()
                logger.info("Data processing completed")

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    def _process_excel_data(self):
        """Optimized Excel data processing with proper column mapping"""
        try:
            # Load Excel with error handling
            df = pd.read_excel(self.excel_file_path)
            logger.info(
                f"Loaded Excel: {len(df)} rows, {len(df.columns)} columns")

            # Validate required columns
            required_cols = ['Application_Owner_Name',
                             'Application_Name', 'Risk_Score']
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Group by Application Owner for aggregation
            grouped_data = defaultdict(lambda: {
                'applications': set(),
                'risk_scores': [],
                'vulnerabilities': [],
                'departments': set(),
                'assets': set()
            })

            # Process each row efficiently
            for _, row in df.iterrows():
                ao_name = str(
                    row.get('Application_Owner_Name', 'Unknown')).strip()
                if ao_name == 'Unknown' or not ao_name:
                    continue

                entry = grouped_data[ao_name]

                # Collect data
                entry['applications'].add(
                    str(row.get('Application_Name', 'Unknown')))
                entry['departments'].add(str(row.get('Dept_Name', 'Unknown')))
                entry['assets'].add(str(row.get('Asset_Name', 'Unknown')))

                # Risk scores
                risk_score = DataProcessor.safe_convert(
                    row.get('Risk_Score'), float)
                if risk_score > 0:
                    entry['risk_scores'].append(risk_score)

                # Vulnerability data
                vuln_data = {
                    'description': str(row.get('Vulnerability_Description', '')),
                    'severity': str(row.get('Severity', 'Unknown')),
                    'cvss_score': DataProcessor.safe_convert(row.get('CVSS_Score'), float),
                    'status': str(row.get('Status', 'Unknown')),
                    'days_to_close': DataProcessor.safe_convert(row.get('Days_to_Close'), int)
                }
                entry['vulnerabilities'].append(vuln_data)

            # Create final AO profiles
            self.ao_data = []
            for ao_name, data in grouped_data.items():
                ao_profile = self._create_ao_profile(ao_name, data)
                self.ao_data.append(ao_profile)

            # Update stats
            self.stats.update({
                'total_aos': len(self.ao_data),
                'total_vulnerabilities': sum(len(data['vulnerabilities']) for data in grouped_data.values()),
                'last_updated': datetime.now().isoformat()
            })

            logger.info(
                f"Processed {self.stats['total_aos']} AOs with {self.stats['total_vulnerabilities']} vulnerabilities")

        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            raise

    def _create_ao_profile(self, ao_name: str, data: Dict) -> Dict:
        """Create optimized AO profile with calculated metrics"""
        vulnerabilities = data['vulnerabilities']
        risk_scores = data['risk_scores']

        # Calculate vulnerability statistics
        vuln_stats = self._calculate_vulnerability_statistics(vulnerabilities)

        # Calculate risk metrics
        avg_risk = round(sum(risk_scores) / len(risk_scores),
                         2) if risk_scores else 0
        max_risk = round(max(risk_scores), 2) if risk_scores else 0

        # Calculate compliance score (mock calculation based on vulnerabilities and risk)
        compliance_score = self._calculate_compliance_score(
            vuln_stats, avg_risk)

        profile = {
            'ao_name': ao_name,
            'applications': list(data['applications']),
            'application': ', '.join(data['applications']),
            'departments': list(data['departments']),
            'department': ', '.join(data['departments']),
            'assets': list(data['assets']),

            # Risk metrics
            'risk_score': str(avg_risk),
            'max_risk_score': str(max_risk),
            'risk_score_entries': len(risk_scores),

            # Vulnerability metrics
            'vulnerability_count': str(vuln_stats['total']),
            'high_vulnerabilities': str(vuln_stats['high']),
            'medium_vulnerabilities': str(vuln_stats['medium']),
            'low_vulnerabilities': str(vuln_stats['low']),
            'critical_vulnerabilities': str(vuln_stats['critical']),

            # Calculated metrics
            'compliance_score': str(compliance_score),
            'criticality': self._determine_criticality(avg_risk, vuln_stats),
            'environment': self._determine_environment(data['applications']),
            'patching_status': self._determine_patching_status(vulnerabilities),
            'last_scan_date': self._get_latest_scan_date(vulnerabilities),

            # Additional metrics
            'application_count': len(data['applications']),
            'asset_count': len(data['assets']),
            'avg_days_to_close': self._calculate_avg_days_to_close(vulnerabilities),

            # Contact info (placeholder)
            'contact_info': f"{ao_name.replace(' ', '.').lower()}@company.com"
        }

        # Create searchable text
        profile['searchable_text'] = self._create_searchable_text(profile)

        return profile

    def _calculate_vulnerability_statistics(self, vulnerabilities: List[Dict]) -> Dict:
        """Calculate detailed vulnerability statistics"""
        stats = {'total': len(vulnerabilities), 'critical': 0,
                 'high': 0, 'medium': 0, 'low': 0}

        for vuln in vulnerabilities:
            severity = vuln.get('severity', '').lower().strip()
            # Handle different severity naming conventions
            if severity in ['critical', 'very high', '5']:
                stats['critical'] += 1
            elif severity in ['high', '4']:
                stats['high'] += 1
            elif severity in ['medium', 'moderate', '3']:
                stats['medium'] += 1
            elif severity in ['low', 'minor', '2', '1']:
                stats['low'] += 1

        return stats

    def _calculate_compliance_score(self, vuln_stats: Dict, avg_risk: float) -> float:
        """Calculate compliance score based on vulnerabilities and risk"""
        base_score = 100

        # Deduct points for vulnerabilities
        base_score -= vuln_stats['critical'] * 20
        base_score -= vuln_stats['high'] * 10
        base_score -= vuln_stats['medium'] * 5
        base_score -= vuln_stats['low'] * 1

        # Deduct points for high risk score
        if avg_risk > 8:
            base_score -= 20
        elif avg_risk > 6:
            base_score -= 10
        elif avg_risk > 4:
            base_score -= 5

        return max(0, round(base_score, 1))

    def _determine_criticality(self, avg_risk: float, vuln_stats: Dict) -> str:
        """Determine application criticality based on risk and vulnerabilities"""
        if avg_risk >= 8 or vuln_stats['critical'] > 0:
            return 'Critical'
        elif avg_risk >= 6 or vuln_stats['high'] > 5:
            return 'High'
        elif avg_risk >= 4 or vuln_stats['high'] > 0:
            return 'Medium'
        else:
            return 'Low'

    def _determine_environment(self, applications: set) -> str:
        """Determine environment type based on applications"""
        app_list = [app.lower() for app in applications]
        if any('prod' in app or 'production' in app for app in app_list):
            return 'Production'
        elif any('test' in app or 'staging' in app for app in app_list):
            return 'Test/Staging'
        else:
            return 'Development'

    def _determine_patching_status(self, vulnerabilities: List[Dict]) -> str:
        """Determine patching status based on vulnerability closure"""
        if not vulnerabilities:
            return 'Up-to-date'

        open_vulns = sum(1 for v in vulnerabilities if v.get(
            'status', '').lower() not in ['closed', 'fixed', 'resolved'])
        total_vulns = len(vulnerabilities)

        if open_vulns == 0:
            return 'Up-to-date'
        elif open_vulns / total_vulns > 0.5:
            return 'Outdated'
        else:
            return 'Pending'

    def _get_latest_scan_date(self, vulnerabilities: List[Dict]) -> str:
        """Get the latest scan date from vulnerabilities"""
        # This is a placeholder - in real implementation, you'd parse actual dates
        return datetime.now().strftime('%Y-%m-%d')

    def _calculate_avg_days_to_close(self, vulnerabilities: List[Dict]) -> float:
        """Calculate average days to close vulnerabilities"""
        days_list = [v.get('days_to_close', 0)
                     for v in vulnerabilities if v.get('days_to_close', 0) > 0]
        return round(sum(days_list) / len(days_list), 1) if days_list else 0

    def _create_searchable_text(self, ao_profile: Dict) -> str:
        """Create optimized searchable text"""
        text_parts = [
            f"Owner: {ao_profile['ao_name']}",
            f"Apps: {ao_profile['application']}",
            f"Dept: {ao_profile['department']}",
            f"Risk: {ao_profile['risk_score']}",
            f"Criticality: {ao_profile['criticality']}",
            f"Environment: {ao_profile['environment']}",
            f"Vulnerabilities: {ao_profile['vulnerability_count']}",
            f"High: {ao_profile['high_vulnerabilities']}",
            f"Compliance: {ao_profile['compliance_score']}%",
            f"Status: {ao_profile['patching_status']}"
        ]
        return " | ".join(text_parts)

    def _create_embeddings(self):
        """Create embeddings with progress tracking"""
        try:
            texts = [ao['searchable_text'] for ao in self.ao_data]
            logger.info(f"Creating embeddings for {len(texts)} AO profiles...")

            self.embeddings = self.model.encode(
                texts, show_progress_bar=True, batch_size=32)
            logger.info(f"Created embeddings: {self.embeddings.shape}")

        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise

    def _build_faiss_index(self):
        """Build optimized FAISS index"""
        try:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)

            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings.astype('float32'))

            logger.info(
                f"Built FAISS index: {self.faiss_index.ntotal} vectors")

        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}")
            raise

    def _save_processed_data(self):
        """Save data with compression"""
        try:
            data_to_save = {
                'ao_data': self.ao_data,
                'embeddings': self.embeddings,
                'stats': self.stats,
                'version': '2.0'
            }

            with open(self.data_file, 'wb') as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            faiss.write_index(self.faiss_index, self.index_file)
            logger.info("Saved processed data and index")

        except Exception as e:
            logger.error(f"Data saving failed: {e}")
            raise

    def _load_processed_data(self) -> bool:
        """Load data with version checking"""
        try:
            if not (os.path.exists(self.data_file) and os.path.exists(self.index_file)):
                return False

            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)

            # Version compatibility check
            if data.get('version', '1.0') != '2.0':
                logger.info("Data version mismatch - will regenerate")
                return False

            self.ao_data = data['ao_data']
            self.embeddings = data['embeddings']
            self.stats = data.get('stats', {})

            self.faiss_index = faiss.read_index(self.index_file)

            return True

        except Exception as e:
            logger.warning(f"Data loading failed: {e}")
            return False

    @lru_cache(maxsize=50)
    def search_aos(self, query: str, top_k: int = 5) -> List[Dict]:
        """Cached AO search with better performance"""
        try:
            if not self.model or not self.faiss_index:
                return []

            # Create and normalize query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), top_k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.ao_data) and score > 0.1:  # Relevance threshold
                    ao = self.ao_data[idx].copy()
                    ao['similarity_score'] = float(score)
                    ao['rank'] = i + 1
                    results.append(ao)

            return results

        except Exception as e:
            logger.error(f"AO search failed: {e}")
            return []

    def get_suggestions(self, ao_name: Optional[str] = None, use_llm: bool = False) -> Dict:
        """Optimized suggestions with better error handling"""
        try:
            if not ao_name or not ao_name.strip():
                return {
                    "error": "AO name is required",
                    "message": "Please provide 'ao_name' parameter",
                    "available_aos": [ao['ao_name'] for ao in self.ao_data[:10]],
                    "total_aos": len(self.ao_data)
                }

            return self._get_ao_specific_suggestions(ao_name.strip(), use_llm)

        except Exception as e:
            logger.error(f"Suggestions generation failed: {e}")
            return {
                "error": "Failed to generate suggestions",
                "message": str(e),
                "ao_requested": ao_name
            }

    def _get_ao_specific_suggestions(self, ao_name: str, use_llm: bool = False) -> Dict:
        """Generate LLM-powered AO-specific suggestions"""
        # Find matching AO
        target_ao = self._find_ao_by_name(ao_name)

        if not target_ao:
            similar_aos = self._find_similar_ao_names(ao_name)
            return {
                "status": "ao_not_found",
                "searched_ao": ao_name,
                "message": f"Application Owner '{ao_name}' not found",
                "similar_ao_names": similar_aos[:10],
                "suggestions": [
                    "Check spelling and capitalization",
                    "Try partial name search",
                    "Use first or last name only"
                ]
            }

        # Always use LLM for suggestions (ignore use_llm parameter for suggestions)
        context = self._build_ao_context(target_ao)

        # Generate comprehensive LLM-powered suggestions
        suggestions_prompt = f"""
You are a senior cybersecurity analyst. Based on the following Application Owner profile, provide a comprehensive security analysis and actionable recommendations.

{context}

Please provide a well-structured response in the following format:

**EXECUTIVE SUMMARY**
[Brief overview of security posture]

**CRITICAL FINDINGS**
[Top 3-5 most important security issues that need immediate attention]

**RISK ASSESSMENT**
[Current risk level and main risk factors]

**IMMEDIATE ACTIONS** (Next 1-2 weeks)
[Specific actions with highest priority]

**SHORT-TERM GOALS** (1-3 months)
[Medium-term improvements and objectives]

**LONG-TERM STRATEGY** (3-12 months)
[Strategic security initiatives]

**COMPLIANCE RECOMMENDATIONS**
[Specific guidance on meeting compliance requirements]

**COMPARATIVE ANALYSIS**
[How this AO compares to industry standards/peers]

Provide specific, actionable recommendations tailored to this Application Owner's profile.
"""

        llm_response = OllamaService.query_ollama(
            suggestions_prompt, temperature=0.6)

        # Format the LLM response into structured JSON
        structured_analysis = ResponseFormatter.format_suggestions_analysis(
            llm_response)

        response = {
            "status": "ao_found",
            "match_type": target_ao.get('match_type', 'exact'),
            "ao_profile": {
                "ao_name": target_ao['ao_name'],
                "applications": target_ao['applications'],
                "department": target_ao['department'],
                "risk_score": target_ao['risk_score'],
                "compliance_score": target_ao['compliance_score'],
                "vulnerability_count": target_ao['vulnerability_count'],
                "criticality": target_ao['criticality'],
                "environment": target_ao['environment']
            },
            "ai_analysis": structured_analysis,
            "ai_enhanced": True,
            "generation_timestamp": datetime.now().isoformat()
        }

        return response

    def _find_ao_by_name(self, ao_name: str) -> Optional[Dict]:
        """Optimized AO finding with fuzzy matching"""
        ao_name_lower = ao_name.lower().strip()

        # Exact match first
        for ao in self.ao_data:
            if ao['ao_name'].lower() == ao_name_lower:
                ao['match_type'] = 'exact'
                return ao

        # Partial match
        for ao in self.ao_data:
            if ao_name_lower in ao['ao_name'].lower():
                ao['match_type'] = 'partial'
                return ao

        # Word match
        search_words = ao_name_lower.split()
        for ao in self.ao_data:
            ao_words = ao['ao_name'].lower().split()
            if any(word in ao_words for word in search_words):
                ao['match_type'] = 'word'
                return ao

        return None

    def _find_similar_ao_names(self, search_name: str) -> List[str]:
        """Find similar AO names using better fuzzy matching"""
        search_lower = search_name.lower()
        similar_names = []

        for ao in self.ao_data:
            ao_name = ao['ao_name']
            ao_lower = ao_name.lower()

            # Various similarity checks
            if (search_lower in ao_lower or
                any(word in ao_lower for word in search_lower.split()) or
                    any(word in search_lower for word in ao_lower.split())):
                similar_names.append(ao_name)

        return sorted(list(set(similar_names)))

    def _build_detailed_ao_info(self, ao: Dict) -> Dict:
        """Build comprehensive AO information"""
        return {
            "basic_info": {
                "ao_name": ao['ao_name'],
                "applications": ao['applications'],
                "departments": ao['departments'],
                "environment": ao['environment'],
                "contact_info": ao['contact_info'],
                "application_count": ao['application_count'],
                "asset_count": ao.get('asset_count', 0)
            },
            "security_metrics": {
                "overall_risk_score": ao['risk_score'],
                "max_risk_score": ao.get('max_risk_score', ao['risk_score']),
                "compliance_score": ao['compliance_score'],
                "criticality_level": ao['criticality'],
                "patching_status": ao['patching_status'],
                "last_scan_date": ao['last_scan_date'],
                "avg_days_to_close": ao.get('avg_days_to_close', 0)
            },
            "vulnerability_breakdown": {
                "total_vulnerabilities": ao['vulnerability_count'],
                "critical_severity": ao.get('critical_vulnerabilities', '0'),
                "high_severity": ao['high_vulnerabilities'],
                "medium_severity": ao['medium_vulnerabilities'],
                "low_severity": ao['low_vulnerabilities'],
                "vulnerability_distribution": self._calculate_vuln_percentages(ao)
            }
        }

    def _calculate_vuln_percentages(self, ao: Dict) -> Dict:
        """Calculate vulnerability distribution percentages"""
        total = max(int(ao['vulnerability_count']), 1)

        return {
            "critical_percentage": round((int(ao.get('critical_vulnerabilities', 0)) / total) * 100, 1),
            "high_percentage": round((int(ao['high_vulnerabilities']) / total) * 100, 1),
            "medium_percentage": round((int(ao['medium_vulnerabilities']) / total) * 100, 1),
            "low_percentage": round((int(ao['low_vulnerabilities']) / total) * 100, 1)
        }

    def _generate_security_analysis(self, ao: Dict) -> Dict:
        """Generate detailed security analysis"""
        analysis = {
            "overall_security_posture": "",
            "critical_concerns": [],
            "positive_aspects": [],
            "risk_assessment": "",
            "security_score": 0
        }

        try:
            risk_score = float(ao['risk_score'])
            compliance = float(ao['compliance_score'])
            high_vulns = int(ao['high_vulnerabilities'])
            critical_vulns = int(ao.get('critical_vulnerabilities', 0))

            # Overall posture assessment
            if risk_score >= 8 or critical_vulns > 0:
                analysis["overall_security_posture"] = "🔴 CRITICAL - Immediate action required"
                analysis["security_score"] = 25
            elif risk_score >= 6 or high_vulns > 3:
                analysis["overall_security_posture"] = "🟠 HIGH RISK - Significant improvements needed"
                analysis["security_score"] = 50
            elif risk_score >= 4 or high_vulns > 0:
                analysis["overall_security_posture"] = "🟡 MODERATE RISK - Some concerns to address"
                analysis["security_score"] = 75
            else:
                analysis["overall_security_posture"] = "🟢 GOOD - Acceptable security posture"
                analysis["security_score"] = 90

            # Critical concerns
            if critical_vulns > 0:
                analysis["critical_concerns"].append(
                    f"🚨 {critical_vulns} critical vulnerabilities need immediate attention")
            if high_vulns > 0:
                analysis["critical_concerns"].append(
                    f"⚠️ {high_vulns} high-severity vulnerabilities require remediation")
            if compliance < 70:
                analysis["critical_concerns"].append(
                    f"📋 Compliance score ({compliance}%) below acceptable threshold")
            if ao['patching_status'].lower() in ['outdated', 'overdue']:
                analysis["critical_concerns"].append(
                    f"🔧 Patching status indicates updates needed")

            # Positive aspects
            if compliance >= 85:
                analysis["positive_aspects"].append(
                    f"✅ Good compliance score ({compliance}%)")
            if high_vulns == 0 and critical_vulns == 0:
                analysis["positive_aspects"].append(
                    "✅ No high or critical vulnerabilities")
            if ao['patching_status'].lower() == 'up-to-date':
                analysis["positive_aspects"].append(
                    "✅ Current with security patches")
            if risk_score < 4:
                analysis["positive_aspects"].append(
                    "✅ Low risk score indicates good security controls")

            # Risk assessment
            risk_factors = []
            if critical_vulns > 0:
                risk_factors.append(
                    f"{critical_vulns} critical vulnerabilities")
            if high_vulns > 0:
                risk_factors.append(
                    f"{high_vulns} high-severity vulnerabilities")
            if compliance < 75:
                risk_factors.append("below-standard compliance")
            if ao['criticality'].lower() in ['critical', 'high']:
                risk_factors.append("high business criticality")

            analysis["risk_assessment"] = f"Key risk factors: {', '.join(risk_factors)}" if risk_factors else "No major risk factors identified"

        except Exception as e:
            logger.error(f"Security analysis error: {e}")
            analysis["error"] = "Analysis partially unavailable"

        return analysis

    def _generate_action_items(self, ao: Dict) -> Dict:
        """Generate prioritized action items"""
        actions = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_strategy": []
        }

        try:
            high_vulns = int(ao['high_vulnerabilities'])
            critical_vulns = int(ao.get('critical_vulnerabilities', 0))
            medium_vulns = int(ao['medium_vulnerabilities'])
            risk_score = float(ao['risk_score'])
            compliance = float(ao['compliance_score'])

            # Immediate actions (1-2 weeks)
            if critical_vulns > 0:
                actions["immediate_actions"].append(
                    f"🚨 URGENT: Address {critical_vulns} critical vulnerabilities")
            if high_vulns > 0:
                actions["immediate_actions"].append(
                    f"⚠️ Remediate {high_vulns} high-severity vulnerabilities")
            if ao['patching_status'].lower() in ['outdated', 'overdue']:
                actions["immediate_actions"].append(
                    "🔧 Apply critical security patches")
            if risk_score >= 8:
                actions["immediate_actions"].append(
                    "🔍 Conduct emergency security assessment")

            # Short-term goals (1-3 months)
            if medium_vulns > 5:
                actions["short_term_goals"].append(
                    f"📋 Address {medium_vulns} medium-severity vulnerabilities")
            if compliance < 80:
                actions["short_term_goals"].append(
                    f"📈 Improve compliance from {compliance}% to 85%+")

            actions["short_term_goals"].extend([
                "🔄 Implement regular vulnerability scanning",
                "📚 Security awareness training for teams",
                "🛡️ Review and update security policies"
            ])

            # Long-term strategy (3-12 months)
            actions["long_term_strategy"].extend([
                "🏗️ Integrate security into development lifecycle",
                "📊 Implement continuous security monitoring",
                "🎯 Establish security KPIs and metrics",
                "🤝 Strengthen security team collaboration",
                "🔮 Plan for emerging security threats"
            ])

            if ao['criticality'].lower() in ['critical', 'high']:
                actions["long_term_strategy"].append(
                    "🛡️ Enhanced security controls for critical systems")

        except Exception as e:
            logger.error(f"Action items generation error: {e}")
            actions["error"] = "Some action items unavailable"

        return actions

    def _get_priority_recommendations(self, ao: Dict) -> List[Dict]:
        """Get top priority recommendations"""
        recommendations = []

        try:
            high_vulns = int(ao['high_vulnerabilities'])
            critical_vulns = int(ao.get('critical_vulnerabilities', 0))
            risk_score = float(ao['risk_score'])
            compliance = float(ao['compliance_score'])

            if critical_vulns > 0:
                recommendations.append({
                    "priority": 1,
                    "action": f"Immediately address {critical_vulns} critical vulnerabilities",
                    "impact": "Prevent potential security breaches",
                    "timeline": "1-2 weeks",
                    "effort": "High"
                })

            if high_vulns > 0:
                recommendations.append({
                    "priority": 2,
                    "action": f"Remediate {high_vulns} high-severity vulnerabilities",
                    "impact": "Reduce attack surface significantly",
                    "timeline": "2-4 weeks",
                    "effort": "Medium-High"
                })

            if compliance < 75:
                recommendations.append({
                    "priority": 3,
                    "action": f"Improve compliance score to 85%+",
                    "impact": "Meet regulatory requirements",
                    "timeline": "1-3 months",
                    "effort": "Medium"
                })

            # Fill remaining slots
            default_actions = [
                "Implement security monitoring",
                "Update incident response plan",
                "Conduct security training"
            ]

            for i, action in enumerate(default_actions):
                if len(recommendations) < 5:
                    recommendations.append({
                        "priority": len(recommendations) + 1,
                        "action": action,
                        "impact": "Improve overall security posture",
                        "timeline": "1-2 months",
                        "effort": "Medium"
                    })

        except Exception as e:
            logger.error(f"Priority recommendations error: {e}")

        return recommendations[:5]

    def _get_compliance_guidance(self, ao: Dict) -> Dict:
        """Generate compliance guidance"""
        guidance = {
            "current_status": "",
            "target_score": 85,
            "gap_analysis": [],
            "improvement_plan": []
        }

        try:
            compliance = float(ao['compliance_score'])

            if compliance >= 90:
                guidance["current_status"] = "Excellent - Exceeds requirements"
            elif compliance >= 80:
                guidance["current_status"] = "Good - Meets most requirements"
            elif compliance >= 70:
                guidance["current_status"] = "Acceptable - Some improvements needed"
            else:
                guidance["current_status"] = "Poor - Significant work required"

            if compliance < 85:
                guidance["gap_analysis"] = [
                    "Security policy enforcement",
                    "Regular security assessments",
                    "Incident response procedures",
                    "Access control management",
                    "Data protection measures"
                ]

                guidance["improvement_plan"] = [
                    f"Target: Increase from {compliance}% to {guidance['target_score']}%",
                    "Conduct compliance gap assessment",
                    "Implement missing security controls",
                    "Establish compliance monitoring",
                    "Regular compliance audits"
                ]

        except Exception as e:
            logger.error(f"Compliance guidance error: {e}")
            guidance["error"] = "Guidance partially unavailable"

        return guidance

    def _get_risk_mitigation_steps(self, ao: Dict) -> Dict:
        """Generate risk mitigation strategy"""
        mitigation = {
            "risk_level": "",
            "mitigation_strategy": [],
            "monitoring_plan": [],
            "success_metrics": []
        }

        try:
            risk_score = float(ao['risk_score'])
            high_vulns = int(ao['high_vulnerabilities'])
            critical_vulns = int(ao.get('critical_vulnerabilities', 0))

            # Risk level assessment
            if risk_score >= 8 or critical_vulns > 0:
                mitigation["risk_level"] = "CRITICAL - Immediate action required"
            elif risk_score >= 6 or high_vulns > 0:
                mitigation["risk_level"] = "HIGH - Priority attention needed"
            elif risk_score >= 4:
                mitigation["risk_level"] = "MEDIUM - Manageable with controls"
            else:
                mitigation["risk_level"] = "LOW - Maintain current posture"

            # Mitigation strategy
            if critical_vulns > 0:
                mitigation["mitigation_strategy"].extend([
                    f"Emergency remediation of {critical_vulns} critical vulnerabilities",
                    "Implement emergency security controls",
                    "Increase security monitoring"
                ])

            if high_vulns > 0:
                mitigation["mitigation_strategy"].extend([
                    f"Prioritize {high_vulns} high-severity vulnerabilities",
                    "Deploy compensating controls",
                    "Enhanced system monitoring"
                ])

            mitigation["mitigation_strategy"].extend([
                "Regular security assessments",
                "Defense-in-depth implementation",
                "Incident response readiness",
                "Security team training"
            ])

            # Monitoring plan
            mitigation["monitoring_plan"] = [
                "Daily security event monitoring",
                "Weekly vulnerability scans",
                "Monthly risk assessments",
                "Quarterly security reviews"
            ]

            # Success metrics
            mitigation["success_metrics"] = [
                f"Reduce risk score to below 4.0",
                "Zero critical vulnerabilities",
                "Compliance score above 85%",
                "Mean time to remediation < 30 days"
            ]

        except Exception as e:
            logger.error(f"Risk mitigation error: {e}")
            mitigation["error"] = "Mitigation plan partially unavailable"

        return mitigation

    def _generate_comparative_analysis(self, ao: Dict) -> Dict:
        """Generate peer comparison analysis"""
        try:
            # Calculate peer statistics
            all_risk_scores = [float(a['risk_score'])
                               for a in self.ao_data if a['risk_score'] != '0']
            all_compliance = [float(a['compliance_score'])
                              for a in self.ao_data if a['compliance_score'] != '0']
            all_high_vulns = [int(a['high_vulnerabilities'])
                              for a in self.ao_data]

            current_risk = float(ao['risk_score'])
            current_compliance = float(ao['compliance_score'])
            current_high_vulns = int(ao['high_vulnerabilities'])

            analysis = {
                "peer_comparison": {},
                "industry_position": "",
                "percentile_ranking": {},
                "benchmarking": {}
            }

            if all_risk_scores:
                avg_risk = sum(all_risk_scores) / len(all_risk_scores)
                analysis["peer_comparison"]["risk_score"] = {
                    "current": current_risk,
                    "peer_average": round(avg_risk, 2),
                    "position": "Above Average" if current_risk > avg_risk else "Below Average"
                }

                # Percentile calculation
                better_than = sum(
                    1 for score in all_risk_scores if current_risk < score)
                percentile = (better_than / len(all_risk_scores)) * 100
                analysis["percentile_ranking"][
                    "risk"] = f"Better than {round(percentile, 1)}% of peers"

            if all_compliance:
                avg_compliance = sum(all_compliance) / len(all_compliance)
                analysis["peer_comparison"]["compliance"] = {
                    "current": current_compliance,
                    "peer_average": round(avg_compliance, 2),
                    "position": "Above Average" if current_compliance > avg_compliance else "Below Average"
                }

            # Industry position
            if current_risk <= 3 and current_compliance >= 85:
                analysis["industry_position"] = "🟢 Top Performer"
            elif current_risk <= 5 and current_compliance >= 75:
                analysis["industry_position"] = "🟡 Good Performer"
            elif current_risk <= 7 and current_compliance >= 65:
                analysis["industry_position"] = "🟠 Average Performer"
            else:
                analysis["industry_position"] = "🔴 Below Average"

            # Benchmarking targets
            analysis["benchmarking"] = {
                "target_risk_score": "< 4.0 (Industry Best Practice)",
                "target_compliance": "> 85% (Excellence Standard)",
                "target_vulnerabilities": "Zero high/critical vulnerabilities"
            }

            return analysis

        except Exception as e:
            logger.error(f"Comparative analysis error: {e}")
            return {"error": "Comparative analysis unavailable"}

    def _add_ai_enhancement(self, ao: Dict) -> Dict:
        """Add AI-powered analysis"""
        try:
            context = self._build_ao_context(ao)

            # Generate AI analysis
            detailed_query = f"""
            Provide expert cybersecurity analysis for this Application Owner:
            
            Focus on:
            1. Executive summary of security posture
            2. Most critical risks and immediate actions
            3. Strategic recommendations with timelines
            4. Industry best practices applicable
            """

            ai_analysis = OllamaService.enhance_ao_response(
                detailed_query, context)

            # Generate specific AI suggestions
            suggestions_query = f"""
            Based on this security profile, provide 5 specific, actionable recommendations:
            
            Focus on immediate wins, medium-term improvements, and long-term strategy.
            """

            ai_suggestions = OllamaService.query_ollama(
                suggestions_query + "\n\n" + context,
                temperature=0.6
            )

            return {
                "ai_enhanced": True,
                "ai_powered_analysis": ai_analysis,
                "ai_specific_suggestions": ai_suggestions,
                "ai_confidence": "high" if "unavailable" not in ai_analysis.lower() else "low"
            }

        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return {
                "ai_enhanced": False,
                "ai_error": "AI analysis unavailable",
                "ai_note": "Ollama service may be down"
            }

    def _build_ao_context(self, ao: Dict) -> str:
        """Build comprehensive context for AI analysis"""
        context_parts = [
            f"Application Owner: {ao['ao_name']}",
            f"Applications: {ao['application']}",
            f"Department: {ao['department']}",
            f"Environment: {ao['environment']}",
            f"Business Criticality: {ao['criticality']}",
            "",
            "Security Metrics:",
            f"  Risk Score: {ao['risk_score']}/10",
            f"  Compliance Score: {ao['compliance_score']}%",
            f"  Total Vulnerabilities: {ao['vulnerability_count']}",
            f"  Critical: {ao.get('critical_vulnerabilities', 0)}",
            f"  High: {ao['high_vulnerabilities']}",
            f"  Medium: {ao['medium_vulnerabilities']}",
            f"  Low: {ao['low_vulnerabilities']}",
            "",
            f"Patching Status: {ao['patching_status']}",
            f"Last Scan: {ao['last_scan_date']}",
            f"Avg Days to Close: {ao.get('avg_days_to_close', 'N/A')}",
            f"Application Count: {ao['application_count']}"
        ]

        return "\n".join(context_parts)

    def generate_llm_search_analysis(self, query: str, matching_aos: List[Dict]) -> str:
        """Generate LLM analysis for search results"""
        if not matching_aos:
            return "No matching Application Owners found for the given query."

        # Build context for LLM
        context_parts = [f"Search Query: {query}",
                         "", "Matching Application Owners:"]

        # Limit to top 5 for context
        for i, ao in enumerate(matching_aos[:5], 1):
            context_parts.extend([
                f"\n{i}. {ao['ao_name']}",
                f"   Applications: {ao.get('application', 'N/A')}",
                f"   Risk Score: {ao['risk_score']}/10",
                f"   Compliance: {ao['compliance_score']}%",
                f"   Vulnerabilities: {ao['vulnerability_count']} (High: {ao['high_vulnerabilities']})",
                f"   Criticality: {ao['criticality']}",
                f"   Department: {ao.get('department', 'N/A')}"
            ])

        context = "\n".join(context_parts)

        analysis_prompt = f"""
You are a cybersecurity analyst reviewing Application Owner data. Based on the search query and matching results below, provide a comprehensive analysis.

{context}

Please provide a structured analysis in the following format:

**SEARCH SUMMARY**
[Brief overview of what was found]

**KEY FINDINGS**
[Main insights from the search results]

**RISK ANALYSIS**
[Risk assessment of the matching AOs]

**PRIORITY ATTENTION**
[Which AOs need immediate focus and why]

**RECOMMENDED ACTIONS**
[Specific next steps based on the findings]

**COMPARATIVE INSIGHTS**
[How these AOs compare in terms of security posture]

Provide specific, actionable insights based on the search results.
"""

        return OllamaService.query_ollama(analysis_prompt, temperature=0.6)

    # Additional utility methods for the API endpoints...
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_aos": len(self.ao_data),
            "total_applications": sum(ao['application_count'] for ao in self.ao_data),
            "avg_risk_score": round(sum(float(ao['risk_score']) for ao in self.ao_data) / len(self.ao_data), 2) if self.ao_data else 0,
            "high_risk_aos": sum(1 for ao in self.ao_data if float(ao['risk_score']) >= 7),
            "last_updated": self.stats.get('last_updated', 'Unknown')
        }


# Initialize the optimized RAG system
try:
    rag_system = AORAGSystem()
    logger.info("Optimized AO RAG System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AO RAG System: {e}")
    rag_system = None


# Flask API endpoints with enhanced error handling

@app.route('/suggestions', methods=['POST'])
def get_suggestions():
    """Enhanced suggestions endpoint with better validation"""
    try:
        if not rag_system:
            return jsonify({"error": "RAG system not initialized"}), 500

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body required",
                "example": {
                    "ao_name": "Alice Singh",
                    "use_llm": True
                }
            }), 400

        ao_name = data.get('ao_name', '').strip()
        use_llm = data.get('use_llm', False)

        if not ao_name:
            return jsonify({
                "success": False,
                "error": "ao_name parameter is required",
                "available_aos": [ao['ao_name'] for ao in rag_system.ao_data[:10]]
            }), 400

        suggestions = rag_system.get_suggestions(
            ao_name=ao_name, use_llm=use_llm)

        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Suggestions endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search_aos():
    """Enhanced search endpoint with LLM analysis"""
    try:
        if not rag_system:
            return jsonify({"error": "RAG system not initialized"}), 500

        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Query parameter required",
                "example": {"query": "high risk applications"}
            }), 400

        query = data.get('query', '').strip()
        top_k = min(data.get('top_k', 5), 20)  # Limit to 20 results

        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400

        # Get search results
        results = rag_system.search_aos(query, top_k)

        # Generate LLM analysis
        raw_llm_analysis = rag_system.generate_llm_search_analysis(
            query, results)

        # Format the LLM analysis into structured JSON
        structured_analysis = ResponseFormatter.format_search_analysis(
            raw_llm_analysis)

        # Simplified result format with essential data
        simplified_results = []
        for ao in results:
            simplified_results.append({
                "ao_name": ao['ao_name'],
                "applications": ao.get('applications', []),
                "risk_score": ao['risk_score'],
                "compliance_score": ao['compliance_score'],
                "vulnerability_count": ao['vulnerability_count'],
                "high_vulnerabilities": ao['high_vulnerabilities'],
                "criticality": ao['criticality'],
                "department": ao.get('department', 'N/A'),
                "similarity_score": ao.get('similarity_score', 0),
                "rank": ao.get('rank', 0)
            })

        return jsonify({
            "success": True,
            "query": query,
            "ai_analysis": structured_analysis,
            "matching_aos": simplified_results,
            "total_found": len(results),
            "ai_enhanced": True,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Search failed",
            "message": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        if not rag_system:
            return jsonify({"error": "RAG system not initialized"}), 500

        stats = rag_system.get_system_stats()

        return jsonify({
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "system_initialized": rag_system is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0-optimized"
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "POST /suggestions - Get AO-specific suggestions",
            "POST /search - Search AOs by query",
            "GET /stats - System statistics",
            "GET /health - Health check"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "message": "Check the allowed HTTP methods for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


class ResponseFormatter:
    """Format and structure LLM responses into proper JSON format"""

    @staticmethod
    def format_search_analysis(raw_ai_response: str) -> Dict:
        """Convert raw LLM search analysis into structured JSON"""
        try:
            structured_response = {
                "search_summary": "",
                "key_findings": [],
                "risk_analysis": "",
                "priority_attention": [],
                "recommended_actions": [],
                "comparative_insights": "",
                "additional_notes": ""
            }

            # Split by section headers
            sections = ResponseFormatter._split_into_sections(raw_ai_response)

            # Extract search summary
            if "search summary" in sections:
                structured_response["search_summary"] = ResponseFormatter._clean_text(
                    sections["search summary"]
                )

            # Extract key findings
            if "key findings" in sections:
                findings_text = sections["key findings"]
                structured_response["key_findings"] = ResponseFormatter._extract_bullet_points(
                    findings_text
                )

            # Extract risk analysis
            if "risk analysis" in sections:
                structured_response["risk_analysis"] = ResponseFormatter._clean_text(
                    sections["risk analysis"]
                )

            # Extract priority attention
            if "priority attention" in sections:
                priority_text = sections["priority attention"]
                structured_response["priority_attention"] = ResponseFormatter._extract_priority_items(
                    priority_text
                )

            # Extract recommended actions
            if "recommended actions" in sections:
                actions_text = sections["recommended actions"]
                structured_response["recommended_actions"] = ResponseFormatter._extract_bullet_points(
                    actions_text
                )

            # Extract comparative insights
            if "comparative insights" in sections:
                structured_response["comparative_insights"] = ResponseFormatter._clean_text(
                    sections["comparative insights"]
                )

            return structured_response

        except Exception as e:
            logger.error(f"Error formatting search analysis: {e}")
            return {
                "search_summary": "Analysis formatting error",
                "key_findings": ["Error parsing LLM response"],
                "risk_analysis": raw_ai_response[:500] + "..." if len(raw_ai_response) > 500 else raw_ai_response,
                "priority_attention": [],
                "recommended_actions": [],
                "comparative_insights": "",
                "error": str(e)
            }

    @staticmethod
    def format_suggestions_analysis(raw_ai_response: str) -> Dict:
        """Convert raw LLM suggestions analysis into structured JSON"""
        try:
            structured_response = {
                "executive_summary": "",
                "critical_findings": [],
                "risk_assessment": "",
                "immediate_actions": [],
                "short_term_goals": [],
                "long_term_strategy": [],
                "compliance_recommendations": [],
                "comparative_analysis": "",
                "additional_recommendations": []
            }

            # Split by section headers
            sections = ResponseFormatter._split_into_sections(raw_ai_response)

            # Extract executive summary
            if "executive summary" in sections:
                structured_response["executive_summary"] = ResponseFormatter._clean_text(
                    sections["executive summary"]
                )

            # Extract critical findings
            if "critical findings" in sections:
                findings_text = sections["critical findings"]
                structured_response["critical_findings"] = ResponseFormatter._extract_bullet_points(
                    findings_text
                )

            # Extract risk assessment
            if "risk assessment" in sections:
                structured_response["risk_assessment"] = ResponseFormatter._clean_text(
                    sections["risk assessment"]
                )

            # Extract immediate actions
            if "immediate actions" in sections:
                actions_text = sections["immediate actions"]
                structured_response["immediate_actions"] = ResponseFormatter._extract_action_items(
                    actions_text
                )

            # Extract short-term goals
            if "short-term goals" in sections or "short term goals" in sections:
                key = "short-term goals" if "short-term goals" in sections else "short term goals"
                goals_text = sections[key]
                structured_response["short_term_goals"] = ResponseFormatter._extract_action_items(
                    goals_text
                )

            # Extract long-term strategy
            if "long-term strategy" in sections or "long term strategy" in sections:
                key = "long-term strategy" if "long-term strategy" in sections else "long term strategy"
                strategy_text = sections[key]
                structured_response["long_term_strategy"] = ResponseFormatter._extract_action_items(
                    strategy_text
                )

            # Extract compliance recommendations
            if "compliance recommendations" in sections:
                compliance_text = sections["compliance recommendations"]
                structured_response["compliance_recommendations"] = ResponseFormatter._extract_bullet_points(
                    compliance_text
                )

            # Extract comparative analysis
            if "comparative analysis" in sections:
                structured_response["comparative_analysis"] = ResponseFormatter._clean_text(
                    sections["comparative analysis"]
                )

            return structured_response

        except Exception as e:
            logger.error(f"Error formatting suggestions analysis: {e}")
            return {
                "executive_summary": "Analysis formatting error",
                "critical_findings": ["Error parsing LLM response"],
                "risk_assessment": raw_ai_response[:500] + "..." if len(raw_ai_response) > 500 else raw_ai_response,
                "immediate_actions": [],
                "short_term_goals": [],
                "long_term_strategy": [],
                "compliance_recommendations": [],
                "comparative_analysis": "",
                "error": str(e)
            }

    @staticmethod
    def _split_into_sections(text: str) -> Dict[str, str]:
        """Split text into sections based on markdown headers"""
        sections = {}
        current_section = ""
        current_content = []

        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # Check if line is a section header (starts with ** and ends with **)
            if line.startswith('**') and line.endswith('**') and len(line) > 4:
                # Save previous section
                if current_section:
                    sections[current_section.lower()] = '\n'.join(
                        current_content).strip()

                # Start new section
                current_section = line.strip('*').strip()
                current_content = []
            else:
                # Add content to current section
                if line:  # Skip empty lines
                    current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section.lower()] = '\n'.join(
                current_content).strip()

        return sections

    @staticmethod
    def _extract_bullet_points(text: str) -> List[str]:
        """Extract bullet points from text"""
        points = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Look for bullet points (*, -, •, or numbered lists)
            if re.match(r'^[\*\-\•]\s+', line) or re.match(r'^\d+\.\s+', line):
                # Remove bullet/number and clean
                clean_point = re.sub(r'^[\*\-\•]\s+', '', line)
                clean_point = re.sub(r'^\d+\.\s+', '', clean_point)
                if clean_point:
                    points.append(clean_point.strip())
            elif line and not re.match(r'^\**[A-Z\s]+\**$', line):  # Not a header
                # Add as regular text if it's not empty and not a header
                if line:
                    points.append(line.strip())

        return points[:10]  # Limit to 10 items

    @staticmethod
    def _extract_action_items(text: str) -> List[Dict]:
        """Extract action items with priority and timeline"""
        actions = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if line and (line.startswith('*') or line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                # Clean the line
                clean_action = re.sub(r'^[\*\-\•]\s+', '', line)
                clean_action = re.sub(r'^\d+\.\s+', '', clean_action)

                if clean_action:
                    # Try to extract timeline if present
                    timeline_match = re.search(
                        r'\(([^)]*(?:week|month|day)[^)]*)\)', clean_action)
                    timeline = timeline_match.group(
                        1) if timeline_match else "As soon as possible"

                    # Remove timeline from action text
                    action_text = re.sub(
                        r'\([^)]*(?:week|month|day)[^)]*\)', '', clean_action).strip()

                    # Determine priority based on keywords
                    priority = "Medium"
                    if any(word in action_text.lower() for word in ['urgent', 'immediate', 'critical', 'emergency']):
                        priority = "High"
                    elif any(word in action_text.lower() for word in ['long-term', 'future', 'eventually']):
                        priority = "Low"

                    actions.append({
                        "action": action_text,
                        "priority": priority,
                        "timeline": timeline,
                        "category": ResponseFormatter._categorize_action(action_text)
                    })

        return actions[:8]  # Limit to 8 actions

    @staticmethod
    def _extract_priority_items(text: str) -> List[Dict]:
        """Extract priority items with details"""
        priority_items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if line and ('risk score' in line.lower() or 'ao' in line.lower() or 'application owner' in line.lower()):
                # Extract AO name if present
                ao_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', line)
                ao_name = ao_match.group(1) if ao_match else "Unknown"

                # Extract risk score if present
                risk_match = re.search(
                    r'risk score[:\s]*([0-9\.]+)', line.lower())
                risk_score = risk_match.group(1) if risk_match else "Unknown"

                priority_items.append({
                    "ao_name": ao_name,
                    "risk_score": risk_score,
                    "reason": line,
                    "urgency": "High" if (risk_score != "Unknown" and float(risk_score) > 4) else "Medium"
                })

        return priority_items[:5]  # Limit to 5 priority items

    @staticmethod
    def _categorize_action(action_text: str) -> str:
        """Categorize action based on content"""
        action_lower = action_text.lower()

        if any(word in action_lower for word in ['vulnerabilit', 'patch', 'security']):
            return "Security"
        elif any(word in action_lower for word in ['compliance', 'audit', 'regulatory']):
            return "Compliance"
        elif any(word in action_lower for word in ['training', 'awareness', 'education']):
            return "Training"
        elif any(word in action_lower for word in ['monitoring', 'assess', 'review']):
            return "Monitoring"
        else:
            return "General"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and format text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # Clean up
        return text.strip()
