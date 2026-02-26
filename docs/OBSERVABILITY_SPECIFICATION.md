# Observability Integration Specification

## Document Information
- **Version**: 1.0
- **Date**: May 22, 2025
- **Project**: Ollama Workbench Observability
- **Primary Platform**: Opik
- **Status**: Implementation Ready

## Table of Contents
1. [Observability Overview](#observability-overview)
2. [Opik Integration Architecture](#opik-integration-architecture)
3. [Implementation Specifications](#implementation-specifications)
4. [Data Collection Strategy](#data-collection-strategy)
5. [Monitoring Dashboards](#monitoring-dashboards)
6. [Alert Management](#alert-management)
7. [Performance Optimization](#performance-optimization)
8. [Privacy and Compliance](#privacy-and-compliance)

---

## Observability Overview

### Strategic Goals
- **Comprehensive Visibility**: 360-degree view into LLM operations, RAG systems, and user workflows
- **Performance Optimization**: Data-driven insights for system and model performance improvements
- **Quality Assurance**: Automated detection of quality degradation and anomalies
- **Cost Management**: Resource utilization tracking and optimization recommendations
- **Predictive Analytics**: Proactive issue detection and capacity planning

### Key Metrics Categories
1. **LLM Performance**: Response time, token throughput, quality scores
2. **RAG Effectiveness**: Retrieval quality, context relevance, answer accuracy
3. **User Experience**: Session duration, feature adoption, error rates
4. **System Health**: Resource utilization, service availability, error rates
5. **Business Metrics**: Usage patterns, cost per operation, user engagement

---

## Opik Integration Architecture

### Platform Selection Rationale
**Primary Choice: Opik**
- Native Ollama integration with minimal configuration overhead
- Open-source model aligns with target market (small businesses, nonprofits)
- Comprehensive LLM observability features out-of-the-box
- Cost-effective solution for 2 Acre Studios' business model
- Professional-grade capabilities without enterprise pricing complexity

### Integration Components

#### 1. Core Opik Integration Layer
```python
# observability/opik_client.py
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from opik import configure, track, opik_context
from opik.api_objects import opik_context as context
from opik.integrations.ollama import OllamaTracker

class OpikIntegration:
    """Central Opik integration for Ollama Workbench."""
    
    def __init__(self):
        self.project_name = "ollama-workbench"
        self.workspace = os.getenv("OPIK_WORKSPACE", "default")
        self.api_key = os.getenv("OPIK_API_KEY")
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            self._configure_opik()
            self.ollama_tracker = OllamaTracker()
    
    def _configure_opik(self):
        """Configure Opik with project settings."""
        configure(
            project_name=self.project_name,
            workspace=self.workspace,
            api_key=self.api_key
        )
    
    @track(
        name="ollama_inference",
        capture_input=True,
        capture_output=True,
        tags=["llm", "inference"]
    )
    async def track_ollama_call(
        self,
        model: str,
        prompt: str,
        response: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Track Ollama model inference with comprehensive metadata."""
        
        # Extract performance metrics
        eval_count = kwargs.get('eval_count', 0)
        eval_duration = kwargs.get('eval_duration', 0)
        total_duration = kwargs.get('total_duration', 0)
        
        # Calculate derived metrics
        tokens_per_second = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
        time_to_first_token = kwargs.get('prompt_eval_duration', 0) / 1e9
        
        # Update current trace with metadata
        with context.update_current_trace(
            metadata={
                "model": model,
                "provider": "ollama",
                "input_tokens": kwargs.get('prompt_eval_count', 0),
                "output_tokens": eval_count,
                "total_tokens": kwargs.get('prompt_eval_count', 0) + eval_count,
                "temperature": kwargs.get('temperature', 0.7),
                "max_tokens": kwargs.get('max_tokens', 150),
                "has_image": kwargs.get('has_image', False),
                "conversation_id": kwargs.get('conversation_id'),
                "user_id": kwargs.get('user_id'),
                "session_id": kwargs.get('session_id')
            },
            tags=["ollama", model.split(":")[0]],
            input_data={"prompt": prompt},
            output_data={"response": response}
        ):
            # Log performance metrics
            context.log_metric("tokens_per_second", tokens_per_second)
            context.log_metric("time_to_first_token", time_to_first_token)
            context.log_metric("total_duration_seconds", total_duration / 1e9)
            context.log_metric("eval_duration_seconds", eval_duration / 1e9)
            
            return {
                "response": response,
                "metadata": {
                    "tokens_per_second": tokens_per_second,
                    "time_to_first_token": time_to_first_token,
                    "total_duration": total_duration,
                    "eval_duration": eval_duration
                }
            }
    
    @track(
        name="rag_pipeline",
        capture_input=True,
        capture_output=True,
        tags=["rag", "retrieval"]
    )
    async def track_rag_query(
        self,
        query: str,
        collection_name: str,
        retrieved_docs: List[Dict],
        reranked_docs: List[Dict],
        final_response: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Track RAG pipeline execution with detailed step analysis."""
        
        with context.update_current_trace(
            metadata={
                "collection": collection_name,
                "query_length": len(query),
                "retrieved_count": len(retrieved_docs),
                "reranked_count": len(reranked_docs),
                "model": kwargs.get('model'),
                "embedding_model": kwargs.get('embedding_model'),
                "reranker_model": kwargs.get('reranker_model')
            },
            input_data={"query": query, "collection": collection_name},
            output_data={"response": final_response}
        ):
            # Track retrieval step
            with context.start_span(name="document_retrieval") as retrieval_span:
                retrieval_span.set_input({"query": query})
                retrieval_span.set_output({
                    "documents": len(retrieved_docs),
                    "avg_score": sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
                })
                context.log_metric("retrieval_count", len(retrieved_docs))
            
            # Track reranking step
            with context.start_span(name="document_reranking") as rerank_span:
                rerank_span.set_input({"documents": len(retrieved_docs)})
                rerank_span.set_output({"reranked_documents": len(reranked_docs)})
                context.log_metric("rerank_count", len(reranked_docs))
            
            # Track generation step
            with context.start_span(name="response_generation") as gen_span:
                gen_span.set_input({
                    "context_docs": len(reranked_docs),
                    "context_length": sum(len(doc.get('content', '')) for doc in reranked_docs)
                })
                gen_span.set_output({"response_length": len(final_response)})
                context.log_metric("response_length", len(final_response))
            
            return {
                "response": final_response,
                "metadata": {
                    "retrieved_count": len(retrieved_docs),
                    "reranked_count": len(reranked_docs),
                    "context_relevance": self._calculate_context_relevance(query, reranked_docs)
                }
            }
    
    @track(
        name="multi_agent_workflow",
        capture_input=True,
        capture_output=True,
        tags=["workflow", "multi-agent"]
    )
    async def track_workflow_execution(
        self,
        workflow_name: str,
        agents: List[str],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Track multi-agent workflow execution."""
        
        with context.update_current_trace(
            metadata={
                "workflow": workflow_name,
                "agent_count": len(agents),
                "agents": agents,
                "complexity": kwargs.get('complexity', 'medium'),
                "estimated_duration": kwargs.get('estimated_duration')
            },
            input_data=input_data,
            output_data=output_data
        ):
            # Track individual agent executions
            for i, agent in enumerate(agents):
                with context.start_span(name=f"agent_{agent}") as agent_span:
                    agent_span.set_metadata({
                        "agent_name": agent,
                        "execution_order": i + 1,
                        "agent_type": kwargs.get(f'{agent}_type', 'unknown')
                    })
            
            context.log_metric("workflow_complexity", len(agents))
            context.log_metric("agent_coordination_overhead", kwargs.get('coordination_time', 0))
            
            return output_data
    
    def _calculate_context_relevance(self, query: str, documents: List[Dict]) -> float:
        """Calculate relevance score for retrieved context."""
        if not documents:
            return 0.0
        
        # Simple relevance calculation based on keyword overlap
        query_terms = set(query.lower().split())
        total_relevance = 0.0
        
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_terms = set(content.split())
            overlap = len(query_terms.intersection(doc_terms))
            relevance = overlap / len(query_terms) if query_terms else 0
            total_relevance += relevance
        
        return total_relevance / len(documents)

# Global instance
opik_client = OpikIntegration()
```

#### 2. Enhanced Ollama Utils Integration
```python
# Enhanced ollama_utils.py with Opik tracing
import ollama
from observability.opik_client import opik_client

async def call_ollama_endpoint(
    model: str,
    prompt: str = None,
    image: str = None,
    conversation_id: str = None,
    user_id: str = None,
    **kwargs
) -> tuple:
    """Enhanced Ollama endpoint call with comprehensive tracing."""
    
    try:
        # Prepare request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        if image:
            request_data["images"] = [image]
            kwargs["has_image"] = True
        
        # Add conversation context
        kwargs.update({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        
        # Execute with timing
        start_time = time.time()
        response = ollama.generate(**request_data)
        end_time = time.time()
        
        # Extract response data
        response_text = response.get("response", "")
        eval_count = response.get("eval_count", 0)
        eval_duration = response.get("eval_duration", 0)
        total_duration = response.get("total_duration", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        prompt_eval_duration = response.get("prompt_eval_duration", 0)
        
        # Calculate metrics
        metrics = {
            "total_time": end_time - start_time,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "total_duration": total_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "tokens_per_second": eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
        }
        
        # Track with Opik
        if opik_client.enabled:
            await opik_client.track_ollama_call(
                model=model,
                prompt=prompt,
                response=response_text,
                **kwargs,
                **metrics
            )
        
        return response_text, response.get("context", []), eval_count, eval_duration, metrics
        
    except Exception as e:
        # Track errors
        if opik_client.enabled:
            with context.update_current_trace(
                tags=["error", "ollama_failure"],
                metadata={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "model": model
                }
            ):
                pass
        raise
```

#### 3. RAG Pipeline Integration
```python
# Enhanced enhanced_rag.py with comprehensive tracking
from observability.opik_client import opik_client

class EnhancedRAGWithObservability:
    """RAG system with comprehensive observability."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_collection(collection_name)
    
    async def query(
        self,
        query: str,
        n_results: int = 5,
        model: str = "llama3:8b",
        conversation_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute RAG query with full observability."""
        
        try:
            # Step 1: Vector retrieval
            retrieval_start = time.time()
            retrieved_docs = await self._retrieve_documents(query, n_results * 2)  # Retrieve more for reranking
            retrieval_time = time.time() - retrieval_start
            
            # Step 2: Reranking
            rerank_start = time.time()
            reranked_docs = await self._rerank_documents(query, retrieved_docs, n_results)
            rerank_time = time.time() - rerank_start
            
            # Step 3: Context preparation
            context = self._prepare_context(reranked_docs)
            
            # Step 4: Response generation
            generation_start = time.time()
            prompt = self._build_prompt(query, context)
            response, _, eval_count, eval_duration, llm_metrics = await call_ollama_endpoint(
                model=model,
                prompt=prompt,
                conversation_id=conversation_id,
                user_id=user_id
            )
            generation_time = time.time() - generation_start
            
            # Track complete RAG pipeline
            if opik_client.enabled:
                await opik_client.track_rag_query(
                    query=query,
                    collection_name=self.collection_name,
                    retrieved_docs=retrieved_docs,
                    reranked_docs=reranked_docs,
                    final_response=response,
                    model=model,
                    embedding_model="all-minilm-l6-v2",  # Default embedding model
                    reranker_model="cross-encoder",
                    retrieval_time=retrieval_time,
                    rerank_time=rerank_time,
                    generation_time=generation_time,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            
            return {
                "response": response,
                "context": reranked_docs,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "rerank_time": rerank_time,
                    "generation_time": generation_time,
                    "total_time": retrieval_time + rerank_time + generation_time,
                    "retrieved_count": len(retrieved_docs),
                    "reranked_count": len(reranked_docs),
                    "context_relevance": self._calculate_relevance_score(query, reranked_docs),
                    **llm_metrics
                }
            }
            
        except Exception as e:
            # Track RAG errors
            if opik_client.enabled:
                with context.update_current_trace(
                    tags=["error", "rag_failure"],
                    metadata={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "query": query,
                        "collection": self.collection_name
                    }
                ):
                    pass
            raise
    
    async def _retrieve_documents(self, query: str, n_results: int) -> List[Dict]:
        """Retrieve documents with timing and quality metrics."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            documents.append({
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance,  # Convert distance to similarity score
                "rank": i + 1
            })
        
        return documents
    
    async def _rerank_documents(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Rerank documents using cross-encoder (simplified implementation)."""
        # This would use a cross-encoder model in production
        # For now, we'll use a simple relevance scoring
        
        scored_docs = []
        for doc in documents:
            relevance_score = self._simple_relevance_score(query, doc["content"])
            doc["rerank_score"] = relevance_score
            scored_docs.append(doc)
        
        # Sort by rerank score and return top_k
        reranked = sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    
    def _simple_relevance_score(self, query: str, document: str) -> float:
        """Simple relevance scoring based on term overlap."""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(doc_terms))
        return overlap / len(query_terms)
    
    def _calculate_relevance_score(self, query: str, documents: List[Dict]) -> float:
        """Calculate average relevance score for the context."""
        if not documents:
            return 0.0
        
        total_score = sum(doc.get("rerank_score", 0) for doc in documents)
        return total_score / len(documents)
```

---

## Implementation Specifications

### Phase 1: Core Integration (Weeks 1-2)

#### Week 1: Environment Setup and Basic Tracing
```python
# Sprint 1 Implementation Plan

# Day 1-2: Environment Setup
def setup_opik_environment():
    """Set up Opik integration environment."""
    
    # 1. Install Opik SDK
    # pip install opik-sdk
    
    # 2. Configure environment variables
    env_vars = {
        "OPIK_API_KEY": "your_opik_api_key",
        "OPIK_WORKSPACE": "ollama-workbench",
        "OPIK_PROJECT": "main",
        "OPIK_ENABLED": "true"
    }
    
    # 3. Initialize Opik client
    from observability.opik_client import opik_client
    
    # 4. Test basic connectivity
    return opik_client.enabled

# Day 3-4: Basic LLM Tracing
@track(name="basic_llm_call")
async def enhanced_ollama_call(model: str, prompt: str, **kwargs):
    """Basic Ollama call with Opik tracing."""
    
    # Track input
    start_time = time.time()
    
    # Execute call
    response = await original_ollama_call(model, prompt, **kwargs)
    
    # Track output and metrics
    duration = time.time() - start_time
    
    with opik_context.update_current_trace(
        metadata={
            "model": model,
            "duration": duration,
            "input_length": len(prompt),
            "output_length": len(response)
        }
    ):
        opik_context.log_metric("response_time", duration)
        opik_context.log_metric("tokens_per_second", len(response.split()) / duration)
    
    return response

# Day 5-7: Integration with Existing Ollama Utils
def integrate_with_ollama_utils():
    """Integrate Opik tracing with existing ollama_utils.py"""
    
    # 1. Modify call_ollama_endpoint function
    # 2. Add comprehensive metadata capture
    # 3. Implement error tracking
    # 4. Add performance metrics
    # 5. Test with various model types
    pass
```

#### Week 2: Enhanced Metadata and Error Handling
```python
# Advanced tracing with comprehensive metadata

class ComprehensiveOllamaTracker:
    """Enhanced Ollama tracking with detailed metrics."""
    
    def __init__(self):
        self.session_cache = {}
        self.model_cache = {}
    
    @track(
        name="comprehensive_ollama_call",
        capture_input=True,
        capture_output=True
    )
    async def track_call(
        self,
        model: str,
        prompt: str,
        conversation_id: str = None,
        user_id: str = None,
        **kwargs
    ):
        """Track Ollama call with comprehensive metadata."""
        
        # Pre-execution setup
        execution_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "model_size": await self._get_model_size(model),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "session_id": kwargs.get("session_id"),
            "prompt_length": len(prompt),
            "prompt_tokens": self._estimate_tokens(prompt),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 150),
            "top_p": kwargs.get("top_p", 0.9),
            "has_system_prompt": "system" in kwargs,
            "has_image": kwargs.get("has_image", False)
        }
        
        # Execution tracking
        try:
            start_time = time.time()
            memory_before = self._get_memory_usage()
            
            # Execute Ollama call
            result = await self._execute_ollama_call(model, prompt, **kwargs)
            
            end_time = time.time()
            memory_after = self._get_memory_usage()
            
            # Post-execution metrics
            execution_metrics = {
                "execution_time": end_time - start_time,
                "memory_delta": memory_after - memory_before,
                "output_length": len(result.get("response", "")),
                "output_tokens": result.get("eval_count", 0),
                "total_tokens": execution_context["prompt_tokens"] + result.get("eval_count", 0),
                "tokens_per_second": result.get("eval_count", 0) / (result.get("eval_duration", 1) / 1e9),
                "time_to_first_token": result.get("prompt_eval_duration", 0) / 1e9,
                "success": True
            }
            
            # Update trace with comprehensive data
            with opik_context.update_current_trace(
                metadata={**execution_context, **execution_metrics},
                tags=self._generate_tags(model, execution_context, execution_metrics),
                input_data={"prompt": prompt},
                output_data={"response": result.get("response", "")}
            ):
                # Log key metrics
                for metric_name, metric_value in execution_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        opik_context.log_metric(metric_name, metric_value)
            
            return result
            
        except Exception as e:
            # Error tracking
            error_context = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
                "success": False
            }
            
            with opik_context.update_current_trace(
                metadata={**execution_context, **error_context},
                tags=["error", "ollama_failure", model.split(":")[0]],
                input_data={"prompt": prompt},
                output_data={"error": str(e)}
            ):
                opik_context.log_metric("error_rate", 1)
            
            raise
    
    async def _get_model_size(self, model: str) -> str:
        """Get model size information."""
        if model not in self.model_cache:
            try:
                info = ollama.show(model)
                self.model_cache[model] = info.get("details", {}).get("parameter_size", "unknown")
            except:
                self.model_cache[model] = "unknown"
        
        return self.model_cache[model]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _generate_tags(self, model: str, context: Dict, metrics: Dict) -> List[str]:
        """Generate relevant tags for the trace."""
        tags = ["ollama", model.split(":")[0]]
        
        # Performance tags
        if metrics.get("tokens_per_second", 0) > 50:
            tags.append("high_performance")
        elif metrics.get("tokens_per_second", 0) < 10:
            tags.append("low_performance")
        
        # Context tags
        if context.get("has_image"):
            tags.append("multimodal")
        
        if context.get("conversation_id"):
            tags.append("conversational")
        
        # Quality tags based on response length
        output_length = metrics.get("output_length", 0)
        if output_length > 1000:
            tags.append("long_response")
        elif output_length < 50:
            tags.append("short_response")
        
        return tags
```

### Phase 2: RAG Pipeline Observability (Weeks 3-4)

#### Advanced RAG Tracking Implementation
```python
# observability/rag_tracker.py

class RAGPipelineTracker:
    """Comprehensive RAG pipeline observability."""
    
    @track(
        name="rag_pipeline_execution",
        capture_input=True,
        capture_output=True,
        tags=["rag", "retrieval", "generation"]
    )
    async def track_rag_pipeline(
        self,
        query: str,
        collection_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Track complete RAG pipeline with step-by-step analysis."""
        
        pipeline_metadata = {
            "collection": collection_name,
            "query_length": len(query),
            "query_complexity": self._analyze_query_complexity(query),
            "retrieval_strategy": kwargs.get("retrieval_strategy", "vector"),
            "embedding_model": kwargs.get("embedding_model", "all-minilm-l6-v2"),
            "reranker_enabled": kwargs.get("reranker_enabled", True),
            "generation_model": kwargs.get("generation_model", "llama3:8b")
        }
        
        with opik_context.update_current_trace(metadata=pipeline_metadata):
            
            # Step 1: Query Processing
            with opik_context.start_span(name="query_processing") as query_span:
                processed_query = await self._process_query(query)
                query_span.set_output({"processed_query": processed_query})
                opik_context.log_metric("query_processing_time", query_span.duration)
            
            # Step 2: Document Retrieval
            with opik_context.start_span(name="document_retrieval") as retrieval_span:
                retrieval_start = time.time()
                retrieved_docs = await self._retrieve_documents(
                    processed_query, 
                    collection_name,
                    **kwargs
                )
                retrieval_time = time.time() - retrieval_start
                
                retrieval_metrics = {
                    "retrieved_count": len(retrieved_docs),
                    "avg_relevance_score": self._calculate_avg_score(retrieved_docs),
                    "retrieval_time": retrieval_time,
                    "collection_size": await self._get_collection_size(collection_name)
                }
                
                retrieval_span.set_output(retrieval_metrics)
                for metric, value in retrieval_metrics.items():
                    opik_context.log_metric(f"retrieval_{metric}", value)
            
            # Step 3: Document Reranking (if enabled)
            reranked_docs = retrieved_docs
            if kwargs.get("reranker_enabled", True):
                with opik_context.start_span(name="document_reranking") as rerank_span:
                    rerank_start = time.time()
                    reranked_docs = await self._rerank_documents(
                        processed_query, 
                        retrieved_docs,
                        **kwargs
                    )
                    rerank_time = time.time() - rerank_start
                    
                    rerank_metrics = {
                        "reranked_count": len(reranked_docs),
                        "rerank_time": rerank_time,
                        "score_improvement": self._calculate_score_improvement(
                            retrieved_docs, reranked_docs
                        )
                    }
                    
                    rerank_span.set_output(rerank_metrics)
                    for metric, value in rerank_metrics.items():
                        opik_context.log_metric(f"rerank_{metric}", value)
            
            # Step 4: Context Preparation
            with opik_context.start_span(name="context_preparation") as context_span:
                context = await self._prepare_context(reranked_docs, **kwargs)
                
                context_metrics = {
                    "context_length": len(context),
                    "context_tokens": self._estimate_tokens(context),
                    "context_documents": len(reranked_docs),
                    "context_density": len(context) / len(reranked_docs) if reranked_docs else 0
                }
                
                context_span.set_output(context_metrics)
                for metric, value in context_metrics.items():
                    opik_context.log_metric(f"context_{metric}", value)
            
            # Step 5: Response Generation
            with opik_context.start_span(name="response_generation") as generation_span:
                generation_start = time.time()
                
                prompt = self._build_rag_prompt(query, context)
                response, _, eval_count, eval_duration, llm_metrics = await call_ollama_endpoint(
                    model=kwargs.get("generation_model", "llama3:8b"),
                    prompt=prompt,
                    **kwargs
                )
                
                generation_time = time.time() - generation_start
                
                generation_metrics = {
                    "generation_time": generation_time,
                    "response_length": len(response),
                    "response_tokens": eval_count,
                    "generation_tokens_per_second": eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0,
                    "context_utilization": self._calculate_context_utilization(context, response)
                }
                
                generation_span.set_output({
                    **generation_metrics,
                    "response": response
                })
                
                for metric, value in generation_metrics.items():
                    opik_context.log_metric(f"generation_{metric}", value)
            
            # Step 6: Quality Assessment
            with opik_context.start_span(name="quality_assessment") as quality_span:
                quality_metrics = await self._assess_response_quality(
                    query, context, response, reranked_docs
                )
                
                quality_span.set_output(quality_metrics)
                for metric, value in quality_metrics.items():
                    opik_context.log_metric(f"quality_{metric}", value)
            
            # Aggregate pipeline metrics
            total_time = retrieval_time + rerank_time + generation_time
            pipeline_efficiency = len(response) / total_time if total_time > 0 else 0
            
            opik_context.log_metric("pipeline_total_time", total_time)
            opik_context.log_metric("pipeline_efficiency", pipeline_efficiency)
            
            return {
                "response": response,
                "context": reranked_docs,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "rerank_time": rerank_time if kwargs.get("reranker_enabled") else 0,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "pipeline_efficiency": pipeline_efficiency,
                    **quality_metrics
                }
            }
    
    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity."""
        word_count = len(query.split())
        
        if word_count < 3:
            return "simple"
        elif word_count < 10:
            return "medium"
        else:
            return "complex"
    
    async def _assess_response_quality(
        self, 
        query: str, 
        context: str, 
        response: str, 
        source_docs: List[Dict]
    ) -> Dict[str, float]:
        """Assess response quality using multiple metrics."""
        
        return {
            "relevance_score": self._calculate_relevance(query, response),
            "context_adherence": self._calculate_context_adherence(context, response),
            "completeness": self._calculate_completeness(query, response),
            "factual_consistency": await self._check_factual_consistency(source_docs, response),
            "readability": self._calculate_readability(response)
        }
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate relevance score between query and response."""
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(response_terms))
        return overlap / len(query_terms)
    
    def _calculate_context_adherence(self, context: str, response: str) -> float:
        """Calculate how well the response adheres to the provided context."""
        context_terms = set(context.lower().split())
        response_terms = set(response.lower().split())
        
        if not context_terms:
            return 0.0
        
        overlap = len(context_terms.intersection(response_terms))
        return overlap / len(context_terms)
    
    def _calculate_completeness(self, query: str, response: str) -> float:
        """Calculate response completeness."""
        # Simple heuristic based on response length relative to query complexity
        query_complexity = len(query.split())
        response_length = len(response.split())
        
        if query_complexity < 5:
            target_length = 50
        elif query_complexity < 15:
            target_length = 150
        else:
            target_length = 300
        
        return min(response_length / target_length, 1.0)
    
    async def _check_factual_consistency(self, source_docs: List[Dict], response: str) -> float:
        """Check factual consistency between sources and response."""
        # This would use a factual consistency model in production
        # For now, return a simple overlap-based metric
        
        if not source_docs:
            return 0.0
        
        source_text = " ".join([doc.get("content", "") for doc in source_docs])
        return self._calculate_context_adherence(source_text, response)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate text readability score."""
        # Simple readability metric based on sentence and word length
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Normalize to 0-1 scale (ideal: 15-20 words per sentence, 4-6 chars per word)
        sentence_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        word_score = 1.0 - abs(avg_word_length - 5) / 5
        
        return (sentence_score + word_score) / 2
```

### Phase 3: Multi-Agent Workflow Tracking (Weeks 5-6)

#### Workflow Orchestration Observability
```python
# observability/workflow_tracker.py

class WorkflowTracker:
    """Track complex multi-agent workflows."""
    
    @track(
        name="multi_agent_workflow",
        capture_input=True,
        capture_output=True,
        tags=["workflow", "multi-agent", "orchestration"]
    )
    async def track_workflow(
        self,
        workflow_name: str,
        workflow_config: Dict[str, Any],
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Track complete multi-agent workflow execution."""
        
        # Initialize workflow tracking
        workflow_metadata = {
            "workflow_name": workflow_name,
            "workflow_type": workflow_config.get("type", "sequential"),
            "agent_count": len(workflow_config.get("agents", [])),
            "estimated_duration": workflow_config.get("estimated_duration", 0),
            "complexity": workflow_config.get("complexity", "medium"),
            "parallel_execution": workflow_config.get("parallel", False)
        }
        
        with opik_context.update_current_trace(metadata=workflow_metadata):
            
            workflow_start = time.time()
            agent_results = {}
            coordination_overhead = 0
            
            # Track each agent execution
            for agent_config in workflow_config.get("agents", []):
                agent_name = agent_config["name"]
                
                with opik_context.start_span(name=f"agent_{agent_name}") as agent_span:
                    
                    # Pre-execution coordination
                    coord_start = time.time()
                    agent_input = await self._prepare_agent_input(
                        agent_config, input_data, agent_results
                    )
                    coordination_overhead += time.time() - coord_start
                    
                    # Agent execution
                    agent_start = time.time()
                    agent_result = await self._execute_agent(
                        agent_name, agent_config, agent_input, **kwargs
                    )
                    agent_duration = time.time() - agent_start
                    
                    # Track agent metrics
                    agent_metrics = {
                        "agent_type": agent_config.get("type", "llm"),
                        "execution_time": agent_duration,
                        "input_tokens": self._estimate_tokens(str(agent_input)),
                        "output_tokens": self._estimate_tokens(str(agent_result)),
                        "success": agent_result.get("success", True),
                        "retries": agent_result.get("retries", 0)
                    }
                    
                    agent_span.set_metadata(agent_metrics)
                    agent_span.set_input(agent_input)
                    agent_span.set_output(agent_result)
                    
                    # Log agent-specific metrics
                    for metric, value in agent_metrics.items():
                        if isinstance(value, (int, float)):
                            opik_context.log_metric(f"agent_{agent_name}_{metric}", value)
                    
                    agent_results[agent_name] = agent_result
            
            # Post-workflow analysis
            workflow_duration = time.time() - workflow_start
            
            # Calculate workflow efficiency metrics
            total_agent_time = sum(
                result.get("execution_time", 0) for result in agent_results.values()
            )
            parallel_efficiency = total_agent_time / workflow_duration if workflow_duration > 0 else 0
            coordination_ratio = coordination_overhead / workflow_duration if workflow_duration > 0 else 0
            
            # Quality assessment
            workflow_quality = await self._assess_workflow_quality(
                workflow_name, input_data, agent_results
            )
            
            # Final workflow metrics
            workflow_metrics = {
                "total_duration": workflow_duration,
                "coordination_overhead": coordination_overhead,
                "parallel_efficiency": parallel_efficiency,
                "coordination_ratio": coordination_ratio,
                "successful_agents": sum(1 for r in agent_results.values() if r.get("success", True)),
                "failed_agents": sum(1 for r in agent_results.values() if not r.get("success", True)),
                "total_retries": sum(r.get("retries", 0) for r in agent_results.values()),
                **workflow_quality
            }
            
            # Log workflow metrics
            for metric, value in workflow_metrics.items():
                if isinstance(value, (int, float)):
                    opik_context.log_metric(f"workflow_{metric}", value)
            
            # Generate final output
            final_output = await self._synthesize_workflow_output(
                workflow_config, agent_results
            )
            
            return {
                "output": final_output,
                "agent_results": agent_results,
                "metadata": workflow_metrics
            }
    
    async def _execute_agent(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        agent_input: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute individual agent with tracking."""
        
        agent_type = agent_config.get("type", "llm")
        
        try:
            if agent_type == "llm":
                return await self._execute_llm_agent(agent_config, agent_input, **kwargs)
            elif agent_type == "tool":
                return await self._execute_tool_agent(agent_config, agent_input, **kwargs)
            elif agent_type == "rag":
                return await self._execute_rag_agent(agent_config, agent_input, **kwargs)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": 0
            }
    
    async def _execute_llm_agent(
        self,
        agent_config: Dict[str, Any],
        agent_input: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute LLM-based agent."""
        
        model = agent_config.get("model", "llama3:8b")
        prompt_template = agent_config.get("prompt_template", "{input}")
        
        # Build prompt
        prompt = prompt_template.format(**agent_input)
        
        # Execute with Ollama
        start_time = time.time()
        response, _, eval_count, eval_duration, metrics = await call_ollama_endpoint(
            model=model,
            prompt=prompt,
            **kwargs
        )
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "output": response,
            "execution_time": execution_time,
            "model": model,
            "input_tokens": metrics.get("prompt_eval_count", 0),
            "output_tokens": eval_count,
            "tokens_per_second": metrics.get("tokens_per_second", 0)
        }
    
    async def _assess_workflow_quality(
        self,
        workflow_name: str,
        input_data: Dict[str, Any],
        agent_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess overall workflow quality."""
        
        # Calculate quality metrics
        success_rate = sum(1 for r in agent_results.values() if r.get("success", True)) / len(agent_results)
        
        avg_response_quality = sum(
            r.get("quality_score", 0.5) for r in agent_results.values()
        ) / len(agent_results) if agent_results else 0
        
        consistency_score = self._calculate_agent_consistency(agent_results)
        
        return {
            "success_rate": success_rate,
            "avg_response_quality": avg_response_quality,
            "consistency_score": consistency_score,
            "workflow_coherence": self._calculate_workflow_coherence(agent_results)
        }
    
    def _calculate_agent_consistency(self, agent_results: Dict[str, Any]) -> float:
        """Calculate consistency between agent outputs."""
        # This would use semantic similarity in production
        # For now, return a placeholder
        return 0.8
    
    def _calculate_workflow_coherence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate overall workflow coherence."""
        # This would analyze the logical flow between agents
        # For now, return a placeholder
        return 0.85
```

---

## Data Collection Strategy

### Data Points and Metrics

#### LLM Interaction Metrics
```python
LLM_METRICS = {
    # Performance Metrics
    "response_time": "float",  # Total response time in seconds
    "tokens_per_second": "float",  # Token generation rate
    "time_to_first_token": "float",  # Latency to first token
    "memory_usage": "float",  # Memory consumption in MB
    "cpu_usage": "float",  # CPU utilization percentage
    
    # Token Metrics
    "input_tokens": "int",  # Number of input tokens
    "output_tokens": "int",  # Number of output tokens
    "total_tokens": "int",  # Total tokens processed
    "context_length": "int",  # Context window used
    
    # Quality Metrics
    "response_length": "int",  # Response length in characters
    "coherence_score": "float",  # Response coherence (0-1)
    "relevance_score": "float",  # Response relevance (0-1)
    "hallucination_score": "float",  # Detected hallucinations (0-1)
    
    # Model Metadata
    "model_name": "string",  # Model identifier
    "model_size": "string",  # Model parameter count
    "provider": "string",  # Model provider (Ollama, OpenAI, etc.)
    "temperature": "float",  # Sampling temperature
    "top_p": "float",  # Top-p sampling parameter
    "max_tokens": "int",  # Maximum token limit
    
    # Context Metadata
    "conversation_id": "string",  # Conversation identifier
    "user_id": "string",  # User identifier
    "session_id": "string",  # Session identifier
    "request_id": "string",  # Unique request identifier
    "timestamp": "datetime",  # Request timestamp
    "has_image": "boolean",  # Multimodal request flag
    "has_system_prompt": "boolean",  # System prompt presence
    
    # Error Tracking
    "error_type": "string",  # Error classification
    "error_message": "string",  # Error details
    "retry_count": "int",  # Number of retries
    "success": "boolean"  # Request success flag
}
```

#### RAG Pipeline Metrics
```python
RAG_METRICS = {
    # Retrieval Metrics
    "retrieval_time": "float",  # Document retrieval time
    "retrieved_count": "int",  # Number of documents retrieved
    "avg_retrieval_score": "float",  # Average relevance score
    "collection_size": "int",  # Total documents in collection
    "embedding_model": "string",  # Embedding model used
    
    # Reranking Metrics
    "rerank_time": "float",  # Reranking execution time
    "reranked_count": "int",  # Final document count
    "score_improvement": "float",  # Reranking effectiveness
    "reranker_model": "string",  # Reranking model used
    
    # Context Metrics
    "context_length": "int",  # Final context length
    "context_tokens": "int",  # Context token count
    "context_density": "float",  # Information density
    "context_utilization": "float",  # How much context was used
    
    # Query Metrics
    "query_length": "int",  # Query length in characters
    "query_complexity": "string",  # Query complexity category
    "query_type": "string",  # Query classification
    
    # Quality Metrics
    "relevance_score": "float",  # Answer relevance (0-1)
    "context_adherence": "float",  # Context faithfulness (0-1)
    "completeness": "float",  # Answer completeness (0-1)
    "factual_consistency": "float",  # Factual accuracy (0-1)
    "citation_accuracy": "float",  # Citation correctness (0-1)
    
    # Pipeline Metrics
    "total_time": "float",  # End-to-end pipeline time
    "pipeline_efficiency": "float",  # Tokens per second overall
    "cache_hit_rate": "float",  # Caching effectiveness
}
```

#### Workflow Orchestration Metrics
```python
WORKFLOW_METRICS = {
    # Execution Metrics
    "workflow_duration": "float",  # Total workflow time
    "agent_count": "int",  # Number of agents
    "parallel_efficiency": "float",  # Parallelization effectiveness
    "coordination_overhead": "float",  # Inter-agent coordination time
    "coordination_ratio": "float",  # Overhead as % of total time
    
    # Agent Metrics
    "successful_agents": "int",  # Agents that completed successfully
    "failed_agents": "int",  # Agents that failed
    "total_retries": "int",  # Total retry attempts
    "avg_agent_time": "float",  # Average agent execution time
    
    # Quality Metrics
    "success_rate": "float",  # Overall workflow success rate
    "consistency_score": "float",  # Inter-agent consistency
    "workflow_coherence": "float",  # Logical flow coherence
    "output_quality": "float",  # Final output quality
    
    # Resource Metrics
    "total_tokens": "int",  # Total tokens across all agents
    "memory_peak": "float",  # Peak memory usage
    "cpu_utilization": "float",  # Average CPU usage
    "network_calls": "int",  # External API calls made
}
```

### Privacy Configuration

#### Data Sensitivity Levels
```python
# observability/privacy_config.py

from enum import Enum
from typing import Dict, Any, Optional

class DataSensitivity(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class PrivacyConfig:
    """Configure data collection based on privacy requirements."""
    
    def __init__(self, privacy_level: str = "standard"):
        self.privacy_level = privacy_level
        self.config = self._load_privacy_config(privacy_level)
    
    def _load_privacy_config(self, level: str) -> Dict[str, Any]:
        """Load privacy configuration based on level."""
        
        configs = {
            "minimal": {
                "collect_prompts": False,
                "collect_responses": False,
                "collect_user_data": False,
                "hash_identifiers": True,
                "retention_days": 7,
                "allowed_fields": [
                    "model_name", "response_time", "token_count",
                    "success", "error_type"
                ]
            },
            
            "standard": {
                "collect_prompts": True,
                "collect_responses": True,
                "collect_user_data": False,
                "hash_identifiers": True,
                "hash_content": False,
                "retention_days": 30,
                "allowed_fields": [
                    "model_name", "response_time", "token_count",
                    "success", "error_type", "prompt_hash",
                    "response_hash", "conversation_id_hash"
                ]
            },
            
            "full": {
                "collect_prompts": True,
                "collect_responses": True,
                "collect_user_data": True,
                "hash_identifiers": False,
                "hash_content": False,
                "retention_days": 90,
                "allowed_fields": "*"  # All fields allowed
            }
        }
        
        return configs.get(level, configs["standard"])
    
    def should_collect_field(self, field_name: str) -> bool:
        """Check if a field should be collected."""
        allowed = self.config.get("allowed_fields", [])
        
        if allowed == "*":
            return True
        
        return field_name in allowed
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data according to privacy configuration."""
        processed = {}
        
        for key, value in data.items():
            if not self.should_collect_field(key):
                continue
            
            # Apply transformations based on privacy settings
            if key in ["user_id", "conversation_id", "session_id"] and self.config.get("hash_identifiers"):
                processed[f"{key}_hash"] = self._hash_value(value)
            elif key in ["prompt", "response"] and self.config.get("hash_content"):
                processed[f"{key}_hash"] = self._hash_value(value)
                processed[f"{key}_length"] = len(str(value))
            elif key in ["prompt", "response"] and not self.config.get(f"collect_{key}s"):
                processed[f"{key}_length"] = len(str(value))
            else:
                processed[key] = value
        
        return processed
    
    def _hash_value(self, value: str) -> str:
        """Hash a value for privacy protection."""
        import hashlib
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]

# Global privacy configuration
privacy_config = PrivacyConfig(
    privacy_level=os.getenv("OBSERVABILITY_PRIVACY_LEVEL", "standard")
)
```

---

## Monitoring Dashboards

### Opik Dashboard Configuration

#### 1. Executive Summary Dashboard
```python
# dashboards/executive_dashboard.py

EXECUTIVE_DASHBOARD_CONFIG = {
    "dashboard_name": "Ollama Workbench - Executive Summary",
    "description": "High-level metrics for business decision making",
    "refresh_interval": "5m",
    "panels": [
        {
            "title": "Active Users (24h)",
            "type": "stat",
            "query": {
                "metric": "unique_users",
                "time_range": "24h",
                "aggregation": "count_distinct"
            },
            "thresholds": {
                "green": 100,
                "yellow": 50,
                "red": 25
            }
        },
        {
            "title": "Total Conversations",
            "type": "stat", 
            "query": {
                "metric": "conversations_total",
                "time_range": "24h",
                "aggregation": "count"
            }
        },
        {
            "title": "Average Response Time",
            "type": "gauge",
            "query": {
                "metric": "response_time",
                "time_range": "24h",
                "aggregation": "avg"
            },
            "unit": "seconds",
            "max": 10
        },
        {
            "title": "Success Rate",
            "type": "gauge",
            "query": {
                "metric": "success_rate",
                "time_range": "24h",
                "aggregation": "avg"
            },
            "unit": "percent",
            "min": 0,
            "max": 100
        },
        {
            "title": "Token Usage Trend",
            "type": "timeseries",
            "query": {
                "metric": "total_tokens",
                "time_range": "7d",
                "aggregation": "sum",
                "group_by": "1h"
            }
        },
        {
            "title": "Model Usage Distribution",
            "type": "pie",
            "query": {
                "metric": "model_usage",
                "time_range": "24h",
                "group_by": "model_name"
            }
        },
        {
            "title": "Error Rate by Type",
            "type": "bar",
            "query": {
                "metric": "error_rate",
                "time_range": "24h",
                "group_by": "error_type"
            }
        },
        {
            "title": "Feature Adoption",
            "type": "heatmap",
            "query": {
                "metric": "feature_usage",
                "time_range": "7d",
                "group_by": ["feature", "day"]
            }
        }
    ]
}
```

#### 2. Technical Performance Dashboard
```python
TECHNICAL_DASHBOARD_CONFIG = {
    "dashboard_name": "Ollama Workbench - Technical Performance",
    "description": "Detailed technical metrics for system optimization",
    "refresh_interval": "1m",
    "panels": [
        {
            "title": "Response Time Distribution",
            "type": "histogram",
            "query": {
                "metric": "response_time",
                "time_range": "1h",
                "buckets": [0.1, 0.5, 1, 2, 5, 10]
            }
        },
        {
            "title": "Tokens per Second by Model",
            "type": "timeseries",
            "query": {
                "metric": "tokens_per_second",
                "time_range": "6h",
                "group_by": "model_name",
                "aggregation": "avg"
            }
        },
        {
            "title": "Memory Usage",
            "type": "timeseries",
            "query": {
                "metric": "memory_usage",
                "time_range": "6h",
                "aggregation": "avg"
            },
            "unit": "MB"
        },
        {
            "title": "RAG Pipeline Performance",
            "type": "table",
            "query": {
                "metrics": [
                    "retrieval_time",
                    "rerank_time", 
                    "generation_time",
                    "total_time",
                    "relevance_score"
                ],
                "time_range": "1h",
                "group_by": "collection_name",
                "aggregation": "avg"
            }
        },
        {
            "title": "Concurrent Users",
            "type": "timeseries",
            "query": {
                "metric": "concurrent_users",
                "time_range": "6h",
                "aggregation": "max"
            }
        },
        {
            "title": "Cache Hit Rate",
            "type": "stat",
            "query": {
                "metric": "cache_hit_rate",
                "time_range": "1h",
                "aggregation": "avg"
            },
            "unit": "percent"
        }
    ]
}
```

#### 3. RAG Quality Dashboard
```python
RAG_QUALITY_DASHBOARD_CONFIG = {
    "dashboard_name": "RAG System Quality",
    "description": "Retrieval-Augmented Generation quality metrics",
    "refresh_interval": "5m",
    "panels": [
        {
            "title": "Answer Relevance Score",
            "type": "gauge",
            "query": {
                "metric": "relevance_score",
                "time_range": "1h",
                "aggregation": "avg"
            },
            "min": 0,
            "max": 1,
            "thresholds": {
                "red": 0.6,
                "yellow": 0.8,
                "green": 0.9
            }
        },
        {
            "title": "Context Adherence",
            "type": "gauge",
            "query": {
                "metric": "context_adherence",
                "time_range": "1h",
                "aggregation": "avg"
            },
            "min": 0,
            "max": 1
        },
        {
            "title": "Factual Consistency",
            "type": "gauge",
            "query": {
                "metric": "factual_consistency",
                "time_range": "1h",
                "aggregation": "avg"
            },
            "min": 0,
            "max": 1
        },
        {
            "title": "Retrieval Quality by Collection",
            "type": "bar",
            "query": {
                "metric": "avg_retrieval_score",
                "time_range": "24h",
                "group_by": "collection_name"
            }
        },
        {
            "title": "Response Completeness",
            "type": "timeseries",
            "query": {
                "metric": "completeness",
                "time_range": "6h",
                "aggregation": "avg"
            }
        },
        {
            "title": "Citation Accuracy",
            "type": "stat",
            "query": {
                "metric": "citation_accuracy",
                "time_range": "1h",
                "aggregation": "avg"
            },
            "unit": "percent"
        }
    ]
}
```

### Custom Alert Rules

#### Performance Alerts
```python
# alerts/performance_alerts.py

PERFORMANCE_ALERTS = [
    {
        "name": "High Response Time",
        "description": "Alert when average response time exceeds threshold",
        "condition": {
            "metric": "response_time",
            "aggregation": "avg",
            "time_window": "5m",
            "threshold": 5.0,
            "operator": ">"
        },
        "severity": "warning",
        "actions": ["email", "slack"],
        "recipients": ["dev-team@company.com"]
    },
    {
        "name": "Low Success Rate",
        "description": "Alert when success rate drops below threshold",
        "condition": {
            "metric": "success_rate",
            "aggregation": "avg",
            "time_window": "10m",
            "threshold": 0.95,
            "operator": "<"
        },
        "severity": "critical",
        "actions": ["email", "slack", "pagerduty"],
        "recipients": ["on-call@company.com"]
    },
    {
        "name": "Memory Usage Spike",
        "description": "Alert when memory usage exceeds safe threshold",
        "condition": {
            "metric": "memory_usage",
            "aggregation": "max",
            "time_window": "5m",
            "threshold": 8000,  # 8GB
            "operator": ">"
        },
        "severity": "warning",
        "actions": ["email", "slack"]
    }
]
```

#### Quality Alerts
```python
QUALITY_ALERTS = [
    {
        "name": "RAG Quality Degradation",
        "description": "Alert when RAG relevance score drops significantly",
        "condition": {
            "metric": "relevance_score",
            "aggregation": "avg",
            "time_window": "15m",
            "threshold": 0.7,
            "operator": "<"
        },
        "severity": "warning",
        "actions": ["email", "slack"],
        "context": {
            "runbook_url": "https://docs.company.com/runbooks/rag-quality",
            "escalation_policy": "rag-team"
        }
    },
    {
        "name": "High Hallucination Rate",
        "description": "Alert when hallucination detection increases",
        "condition": {
            "metric": "hallucination_score",
            "aggregation": "avg",
            "time_window": "30m",
            "threshold": 0.3,
            "operator": ">"
        },
        "severity": "critical",
        "actions": ["email", "slack", "incident"],
        "auto_actions": ["disable_affected_models"]
    }
]
```

---

This observability specification provides comprehensive guidance for implementing world-class monitoring and analytics for Ollama Workbench using Opik as the primary platform. The phased implementation approach ensures steady progress while maintaining system stability and performance.