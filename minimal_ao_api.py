"""
Minimal AO RAG API - Single File Implementation
Contains only 2 POST routes: /suggestions and /assistant
All RAG functionality embedded in this file.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json
import requests
import asyncio
import threading
from typing import Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaService:
    @staticmethod
    def query_ollama(prompt: str, temperature: float = 0.7, model: str = "llama3.1:8b") -> str:
        """Query Ollama LLM with the given prompt"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,  # Get complete response for Flask integration
                "options": {"temperature": temperature}
            }

            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data.get('response', 'No response from Ollama')

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return f"Error: Unable to connect to Ollama service. Please ensure Ollama is running on localhost:11434"
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return f"Error: {str(e)}"

    @staticmethod
    def analyze_vulnerability(vulnerability_name: str, code_snippet: str = "", risk_rating: str = "", description: str = "") -> str:
        """Analyze vulnerability using Ollama"""
        if vulnerability_name and not code_snippet and not risk_rating:
            # OWASP categorization mode
            normalized_vuln = vulnerability_name.replace(
                '-', ' ').strip().title()
            prompt = f"""
You are a security expert. Given the vulnerability name and description below, first group it under its main/umbrella vulnerability category, then predict the most relevant OWASP Top 10 category for the umbrella group, using ONLY the official OWASP Top 10 (2021) categories listed below as reference.

Here are some examples:
- Vulnerability Name: Blind SQL Injection
  Umbrella Category: SQL Injection
  OWASP Category: A03:2021 – Injection

- Vulnerability Name: PHP Object Injection
  Umbrella Category: Insecure Deserialization
  OWASP Category: A08:2021 – Software and Data Integrity Failures

- Vulnerability Name: Reflected XSS
  Umbrella Category: Cross-Site Scripting (XSS)
  OWASP Category: A07:2021 – Identification and Authentication Failures

- Vulnerability Name: Broken Authentication
  Umbrella Category: Broken Authentication
  OWASP Category: A07:2021 – Identification and Authentication Failures

- Vulnerability Name: Server-Side Request Forgery
  Umbrella Category: Server-Side Request Forgery (SSRF)
  OWASP Category: A10:2021 – Server-Side Request Forgery (SSRF)

Now, use the same logic for the following:

OWASP Top 10 (2021) categories:
A01:2021 – Broken Access Control
A02:2021 – Cryptographic Failures
A03:2021 – Injection
A04:2021 – Insecure Design
A05:2021 – Security Misconfiguration
A06:2021 – Vulnerable and Outdated Components
A07:2021 – Identification and Authentication Failures
A08:2021 – Software and Data Integrity Failures
A09:2021 – Security Logging and Monitoring Failures
A10:2021 – Server-Side Request Forgery (SSRF)

Vulnerability Name: {normalized_vuln}
Description: {description}
"""
            return OllamaService.query_ollama(prompt, temperature=0.5)
        else:
            # Code analysis mode
            prompt = f"""
You are a security expert. Analyze the following code for the vulnerability described below.

Vulnerability Name: {vulnerability_name}
Risk Rating: {risk_rating}

Code Snippet:
{code_snippet}

For this vulnerability, provide:
1. A description of the vulnerability.
2. An explanation of how it occurs in the code.
3. A patch recommendation to fix it.
"""
            return OllamaService.query_ollama(prompt, temperature=0.7)

    @staticmethod
    def enhance_ao_response(query: str, ao_context: str) -> str:
        """Enhance AO response with LLM analysis"""
        prompt = f"""
You are a cybersecurity analyst expert. Based on the Application Owner (AO) data provided below, give a comprehensive security analysis and actionable recommendations.

User Query: {query}

Application Owner Data:
{ao_context}

Please provide:
1. A summary of the security posture
2. Key risk areas and vulnerabilities
3. Prioritized recommendations for improvement
4. Compliance considerations
5. Immediate action items

Please answer in English and be specific to the data provided.
"""
        return OllamaService.query_ollama(prompt, temperature=0.6)


class AORAGSystem:
    def __init__(self, excel_file_path="Cybersecurity_KPI_Minimal.xlsx"):
        self.excel_file_path = excel_file_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ao_data = []
        self.embeddings = None
        self.faiss_index = None
        self.data_file = "ao_rag_data.pkl"
        self.index_file = "ao_rag_faiss.index"

        # Initialize the system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize or load the RAG system"""
        try:
            if self._load_processed_data():
                logger.info("Loaded existing processed data and FAISS index")
            else:
                logger.info("Processing data from scratch...")
                self._process_excel_data()
                self._create_embeddings()
                self._build_faiss_index()
                self._save_processed_data()
                logger.info("Data processing completed and saved")
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise

    def _process_excel_data(self):
        """Process Excel data and create AO profiles"""
        try:
            df = pd.read_excel(self.excel_file_path)
            logger.info(
                f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")

            self.ao_data = []
            for index, row in df.iterrows():
                # Create comprehensive AO profile
                ao_profile = {
                    'ao_name': str(row.get('Application Owner', 'Unknown')),
                    'application': str(row.get('Application', 'Unknown')),
                    'criticality': str(row.get('Criticality', 'Unknown')),
                    'environment': str(row.get('Environment', 'Unknown')),
                    'vulnerability_count': str(row.get('Vulnerability Count', 0)),
                    'high_vulnerabilities': str(row.get('High Vulnerabilities', 0)),
                    'medium_vulnerabilities': str(row.get('Medium Vulnerabilities', 0)),
                    'low_vulnerabilities': str(row.get('Low Vulnerabilities', 0)),
                    'patching_status': str(row.get('Patching Status', 'Unknown')),
                    'compliance_score': str(row.get('Compliance Score', 0)),
                    'last_scan_date': str(row.get('Last Scan Date', 'Unknown')),
                    'risk_score': str(row.get('Risk Score', 0)),
                    'department': str(row.get('Department', 'Unknown')),
                    'contact_info': str(row.get('Contact Info', 'Unknown')),
                    'index': index
                }

                # Create searchable text
                searchable_text = self._create_searchable_text(ao_profile)
                ao_profile['searchable_text'] = searchable_text

                self.ao_data.append(ao_profile)

            logger.info(f"Processed {len(self.ao_data)} AO profiles")

        except Exception as e:
            logger.error(f"Error processing Excel data: {e}")
            raise

    def _create_searchable_text(self, ao_profile):
        """Create searchable text for an AO profile"""
        text_parts = [
            f"Application Owner: {ao_profile['ao_name']}",
            f"Application: {ao_profile['application']}",
            f"Criticality: {ao_profile['criticality']}",
            f"Environment: {ao_profile['environment']}",
            f"Department: {ao_profile['department']}",
            f"Total Vulnerabilities: {ao_profile['vulnerability_count']}",
            f"High Risk Vulnerabilities: {ao_profile['high_vulnerabilities']}",
            f"Medium Risk Vulnerabilities: {ao_profile['medium_vulnerabilities']}",
            f"Low Risk Vulnerabilities: {ao_profile['low_vulnerabilities']}",
            f"Patching Status: {ao_profile['patching_status']}",
            f"Compliance Score: {ao_profile['compliance_score']}",
            f"Risk Score: {ao_profile['risk_score']}",
            f"Last Scan: {ao_profile['last_scan_date']}",
            f"Contact: {ao_profile['contact_info']}"
        ]
        return " | ".join(text_parts)

    def _create_embeddings(self):
        """Create embeddings for all AO profiles"""
        try:
            texts = [ao['searchable_text'] for ao in self.ao_data]
            logger.info("Creating embeddings...")
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(
                f"Created embeddings with shape: {self.embeddings.shape}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            dimension = self.embeddings.shape[1]
            # Inner product for cosine similarity
            self.faiss_index = faiss.IndexFlatIP(dimension)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings.astype('float32'))

            logger.info(
                f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def _save_processed_data(self):
        """Save processed data and FAISS index"""
        try:
            # Save data
            with open(self.data_file, 'wb') as f:
                pickle.dump({
                    'ao_data': self.ao_data,
                    'embeddings': self.embeddings
                }, f)

            # Save FAISS index
            faiss.write_index(self.faiss_index, self.index_file)
            logger.info("Saved processed data and FAISS index")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

    def _load_processed_data(self):
        """Load processed data and FAISS index"""
        try:
            if not (os.path.exists(self.data_file) and os.path.exists(self.index_file)):
                return False

            # Load data
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.ao_data = data['ao_data']
                self.embeddings = data['embeddings']

            # Load FAISS index
            self.faiss_index = faiss.read_index(self.index_file)

            return True
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return False

    def search_aos(self, query, top_k=5):
        """Search for AOs based on query"""
        try:
            # Create query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search using FAISS
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), top_k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.ao_data):
                    ao = self.ao_data[idx].copy()
                    ao['similarity_score'] = float(score)
                    ao['rank'] = i + 1
                    results.append(ao)

            return results
        except Exception as e:
            logger.error(f"Error searching AOs: {e}")
            return []

    def get_suggestions(self):
        """Generate suggestions for AO queries"""
        try:
            # Analyze the data to provide meaningful suggestions
            total_aos = len(self.ao_data)

            # Get unique values for suggestions
            applications = list(set(
                [ao['application'] for ao in self.ao_data if ao['application'] != 'Unknown']))[:5]
            departments = list(set(
                [ao['department'] for ao in self.ao_data if ao['department'] != 'Unknown']))[:5]

            # Calculate some statistics
            high_risk_aos = [ao for ao in self.ao_data if int(
                ao['high_vulnerabilities']) > 0]
            critical_apps = [ao for ao in self.ao_data if ao['criticality'].lower() in [
                'high', 'critical']]

            suggestions = {
                "query_suggestions": [
                    "Show me AOs with high vulnerabilities",
                    "Find application owners in production environment",
                    "Who are the AOs with critical applications?",
                    "Show me AOs with poor compliance scores",
                    "Find AOs who need immediate patching",
                    "Which AOs have the highest risk scores?",
                    "Show me recent vulnerability scan results",
                    "Find AOs in specific departments"
                ],
                "application_highlights": applications,
                "department_highlights": departments,
                "statistics": {
                    "total_aos": total_aos,
                    "high_risk_aos": len(high_risk_aos),
                    "critical_applications": len(critical_apps),
                    "avg_vulnerabilities": round(sum([int(ao['vulnerability_count']) for ao in self.ao_data]) / total_aos, 2) if total_aos > 0 else 0
                },
                "priority_areas": [
                    f"{len(high_risk_aos)} AOs with high vulnerabilities need attention",
                    f"{len(critical_apps)} critical applications require monitoring",
                    "Regular patching status updates recommended",
                    "Compliance score improvements needed"]
            }

            return suggestions
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return {"error": "Failed to generate suggestions"}

    def get_assistant_response(self, query, use_llm=True):
        """Generate assistant response with context and optional LLM enhancement"""
        try:
            # Search for relevant AOs
            search_results = self.search_aos(query, top_k=3)

            if not search_results:
                return {
                    "response": "I couldn't find any relevant Application Owners for your query. Please try rephrasing your question or use the suggestions endpoint for query ideas.",
                    "context": [],
                    "recommendations": ["Try broader search terms", "Check the suggestions endpoint for query ideas"],
                    "llm_enhanced": False
                }

            # Generate context-rich response
            context_data = []
            response_parts = []

            response_parts.append(
                f"Based on your query '{query}', I found {len(search_results)} relevant Application Owner(s):")

            # Build context string for LLM
            llm_context_parts = []

            for i, ao in enumerate(search_results, 1):
                context_data.append({
                    "rank": i,
                    "ao_name": ao['ao_name'],
                    "application": ao['application'],
                    "criticality": ao['criticality'],
                    "vulnerability_summary": {
                        "total": ao['vulnerability_count'],
                        "high": ao['high_vulnerabilities'],
                        "medium": ao['medium_vulnerabilities'],
                        "low": ao['low_vulnerabilities']
                    },
                    "risk_metrics": {
                        "risk_score": ao['risk_score'],
                        "compliance_score": ao['compliance_score'],
                        "patching_status": ao['patching_status']
                    },
                    "environment": ao['environment'],
                    "department": ao['department'],
                    "last_scan_date": ao['last_scan_date'],
                    "similarity_score": round(ao['similarity_score'], 3)
                })

                response_parts.append(
                    f"\n{i}. {ao['ao_name']} - {ao['application']}")
                response_parts.append(f"   • Criticality: {ao['criticality']}")
                response_parts.append(
                    f"   • Vulnerabilities: {ao['vulnerability_count']} total ({ao['high_vulnerabilities']} high risk)")
                response_parts.append(f"   • Risk Score: {ao['risk_score']}")
                response_parts.append(
                    f"   • Compliance: {ao['compliance_score']}")
                response_parts.append(f"   • Department: {ao['department']}")

                # Build LLM context
                llm_context_parts.append(f"""
AO {i}: {ao['ao_name']}
- Application: {ao['application']}
- Criticality: {ao['criticality']}
- Environment: {ao['environment']}
- Department: {ao['department']}
- Total Vulnerabilities: {ao['vulnerability_count']}
- High Risk Vulnerabilities: {ao['high_vulnerabilities']}
- Medium Risk Vulnerabilities: {ao['medium_vulnerabilities']}
- Low Risk Vulnerabilities: {ao['low_vulnerabilities']}
- Risk Score: {ao['risk_score']}
- Compliance Score: {ao['compliance_score']}
- Patching Status: {ao['patching_status']}
- Last Scan Date: {ao['last_scan_date']}
""")

            # Generate recommendations
            recommendations = self._generate_recommendations(search_results)

            base_response = "\n".join(response_parts)
            llm_analysis = ""

            # Enhanced LLM response if requested
            if use_llm:
                try:
                    llm_context = "\n".join(llm_context_parts)
                    llm_analysis = OllamaService.enhance_ao_response(
                        query, llm_context)
                except Exception as e:
                    logger.error(f"LLM enhancement failed: {e}")
                    llm_analysis = "LLM enhancement unavailable"

            response = {
                "response": base_response,
                "llm_analysis": llm_analysis if use_llm else None,
                "context": context_data,
                "recommendations": recommendations,
                "query_processed": query,
                "total_matches": len(search_results),
                "timestamp": datetime.now().isoformat(),
                "llm_enhanced": use_llm and llm_analysis != "LLM enhancement unavailable"
            }

            return response

        except Exception as e:
            logger.error(f"Error generating assistant response: {e}")
            return {
                "response": "An error occurred while processing your request.",
                "context": [],
                "recommendations": ["Please try again or contact support"],
                "llm_enhanced": False
            }

    def _generate_recommendations(self, search_results):
        """Generate actionable recommendations based on search results"""
        recommendations = []

        try:
            # Analyze the results
            high_vuln_aos = [ao for ao in search_results if int(
                ao['high_vulnerabilities']) > 0]
            critical_apps = [ao for ao in search_results if ao['criticality'].lower() in [
                'high', 'critical']]
            poor_compliance = [ao for ao in search_results if float(
                ao['compliance_score']) < 70]

            if high_vuln_aos:
                recommendations.append(
                    f"Priority: {len(high_vuln_aos)} AO(s) have high-risk vulnerabilities requiring immediate attention")

            if critical_apps:
                recommendations.append(
                    f"Monitor: {len(critical_apps)} critical application(s) need enhanced security monitoring")

            if poor_compliance:
                recommendations.append(
                    f"Compliance: {len(poor_compliance)} AO(s) need compliance score improvement")

            # General recommendations
            recommendations.extend([
                "Review patching schedules for identified AOs",
                "Coordinate with AOs for vulnerability remediation timelines",
                "Schedule follow-up security assessments"
            ])

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = [
                "Review the identified AOs for security improvements"]

        return recommendations


# Initialize the RAG system
try:
    rag_system = AORAGSystem()
    logger.info("AO RAG System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AO RAG System: {e}")
    rag_system = None


@app.route('/suggestions', methods=['POST'])
def get_suggestions():
    """
    POST /suggestions
    Get smart suggestions for AO queries and system insights
    """
    try:
        if not rag_system:
            return jsonify({"error": "RAG system not initialized"}), 500

        suggestions = rag_system.get_suggestions()

        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
            "message": "Here are some suggestions to help you explore the AO data"
        }), 200

    except Exception as e:
        logger.error(f"Error in suggestions endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to generate suggestions",
            "message": str(e)
        }), 500


@app.route('/assistant', methods=['POST'])
def get_assistant():
    """
    POST /assistant
    Main assistant endpoint for AO queries with context generation and optional LLM enhancement
    """
    try:
        if not rag_system:
            return jsonify({"error": "RAG system not initialized"}), 500

        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required field 'query' in request body",
                "example": {"query": "Show me AOs with high vulnerabilities", "use_llm": True}
            }), 400

        query = data['query'].strip()
        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400

        # Check if LLM enhancement is requested (default: True)
        use_llm = data.get('use_llm', True)

        # Get assistant response with optional LLM enhancement
        response = rag_system.get_assistant_response(query, use_llm=use_llm)

        return jsonify({
            "success": True,
            "assistant_response": response,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in assistant endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to process assistant request",
            "message": str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    POST /chat
    Analyze vulnerabilities and code using Ollama LLM
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Missing request body",
                "example": {
                    "vulnerability_name": "SQL Injection",
                    "description": "Optional vulnerability description",
                    "code_snippet": "Optional code to analyze",
                    "risk_rating": "High"
                }
            }), 400

        vulnerability_name = data.get('vulnerability_name', '')
        code_snippet = data.get('code_snippet', '')
        risk_rating = data.get('risk_rating', '')
        description = data.get('description', '')

        if not vulnerability_name:
            return jsonify({
                "success": False,
                "error": "vulnerability_name is required",
                "example": {"vulnerability_name": "SQL Injection"}
            }), 400

        # Analyze vulnerability with Ollama
        analysis = OllamaService.analyze_vulnerability(
            vulnerability_name=vulnerability_name,
            code_snippet=code_snippet,
            risk_rating=risk_rating,
            description=description
        )

        return jsonify({
            "success": True,
            "vulnerability_analysis": analysis,
            "input": {
                "vulnerability_name": vulnerability_name,
                "code_snippet": code_snippet,
                "risk_rating": risk_rating,
                "description": description
            },
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to analyze vulnerability",
            "message": str(e)
        }), 500


@app.route('/direct', methods=['POST'])
def direct():
    """
    POST /direct
    Direct chat with Ollama LLM
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required field 'query' in request body",
                "example": {"query": "What are the most common web application vulnerabilities?"}
            }), 400

        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400

        # Ensure response is in English
        if 'please answer in english' not in user_query.lower():
            user_query = user_query.strip() + "\n\nPlease answer in English."

        # Query Ollama directly
        response = OllamaService.query_ollama(user_query, temperature=0.8)

        return jsonify({
            "success": True,
            "llm_response": response,
            "query": user_query,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in direct endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to process direct query",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "POST /suggestions - Get smart suggestions for AO queries",
            "POST /assistant - Ask questions about Application Owners (with optional LLM enhancement)",
            "POST /chat - Analyze vulnerabilities and code using Ollama LLM",
            "POST /direct - Direct chat with Ollama LLM"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "message": "Only POST methods are supported for this API"
    }), 405


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ENHANCED AO RAG API WITH OLLAMA INTEGRATION")
    print("="*70)
    print("📋 Available Endpoints:")
    print("   POST /suggestions  - Get smart suggestions")
    print("   POST /assistant    - Ask AO questions (with LLM enhancement)")
    print("   POST /chat         - Analyze vulnerabilities with Ollama")
    print("   POST /direct       - Direct chat with Ollama LLM")
    print("="*70)
    print("💡 Example Usage:")
    print("   POST http://localhost:5001/suggestions")
    print("   POST http://localhost:5001/assistant")
    print(
        "   Body: {\"query\": \"Show me AOs with high vulnerabilities\", \"use_llm\": true}")
    print("   POST http://localhost:5001/chat")
    print(
        "   Body: {\"vulnerability_name\": \"SQL Injection\", \"code_snippet\": \"...\"}")
    print("   POST http://localhost:5001/direct")
    print("   Body: {\"query\": \"Explain OWASP Top 10\"}")
    print("="*70)

    app.run(debug=True, host='0.0.0.0', port=5001)
