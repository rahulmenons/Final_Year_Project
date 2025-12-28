from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.db import transaction

from .models import Document, Keyword, DocumentKeyword
from .serializers import DocumentSerializer, FileUploadSerializer
from .services import (
    DocumentParser,
    KeywordExtractor,
    DocumentSummarizer,
    evaluate_and_save,   # ✅ NEW IMPORT
    RFPMetadataExtractor,
)


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    @action(detail=False, methods=['post'], url_path='upload')
    def upload_document(self, request):
        """Upload and process document to extract keywords, summary, and evaluation"""
        serializer = FileUploadSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                serializer.errors, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = serializer.validated_data['file']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # 1) Parse document
            print(f"Parsing {uploaded_file.name}...")
            parser = DocumentParser()
            text = parser.parse(uploaded_file, file_extension)
            
            if not text:
                return Response(
                    {'error': 'Could not extract text from document'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            print(f"Extracted {len(text)} characters of text")
            
            # 2) Extract keywords
            print("Extracting keywords...")
            extractor = KeywordExtractor()
            keywords_with_scores = extractor.extract_keywords(text, top_n=15)
            print(f"Found {len(keywords_with_scores)} keywords")
            
            # 3) Generate summary
            print("Generating summary...")
            summarizer = DocumentSummarizer()
            summary_text = summarizer.generate_summary(text, max_length=350)
            print(f"Summary generated: {len(summary_text)} characters")

            # 4) NEW: Extract RFP metadata (budget, timeline, team size)
            print("Extracting RFP metadata (budget/timeline/team size)...")
            metadata_extractor = RFPMetadataExtractor()
            meta = metadata_extractor.extract_metadata(text)
            print("🎯 EXTRACTED RFP METADATA (from Gemini)")
            print(meta)
            print(f"Metadata extracted: {meta}")
            
            rfp_budget = meta.get("budget_in_inr") or 0
            rfp_timeline_weeks = meta.get("timeline_weeks") or 0
            rfp_team_size_required = meta.get("team_size_required") or 0

            print(f"Budget extracted: {rfp_budget}")
            print(f"Timeline extracted: {rfp_timeline_weeks}")
            print(f"Team size extracted: {rfp_team_size_required}")

            
            # 5) Save to database
            with transaction.atomic():
                document = Document.objects.create(
                    filename=uploaded_file.name,
                    file=uploaded_file,
                    file_type=file_extension,
                    content_preview=text[:1000],
                    summary=summary_text,
                    processed=False,  # will be set True in evaluate_and_save
                    rfp_budget=rfp_budget,
                    rfp_timeline_weeks=rfp_timeline_weeks,
                    rfp_team_size_required=rfp_team_size_required,
                )
                
                for keyword_text, score in keywords_with_scores:
                    keyword, created = Keyword.objects.get_or_create(
                        keyword=keyword_text
                    )
                    DocumentKeyword.objects.create(
                        document=document,
                        keyword=keyword,
                        relevance_score=float(score)
                    )

                # 6) Evaluate RFP vs company capability
                evaluation = evaluate_and_save(document)
            
            response_serializer = DocumentSerializer(document)
            data = response_serializer.data
            data["evaluation"] = {
                "technical_fit_score": evaluation.technical_fit_score,
                "budget_fit_score": evaluation.budget_fit_score,
                "timeline_fit_score": evaluation.timeline_fit_score,
                "capacity_fit_score": evaluation.capacity_fit_score,
                "overall_fit_score": evaluation.overall_fit_score,
                "decision": evaluation.decision,
                "reasoning": evaluation.reasoning,
            }
            # Include raw extracted meta as well if you want to show in UI
            data["rfp_metadata"] = {
                "budget_in_inr": rfp_budget,
                "timeline_weeks": rfp_timeline_weeks,
                "team_size_required": rfp_team_size_required,
                "extraction_confidence": meta.get("confidence"),
                "extraction_notes": meta.get("notes"),
            }

            return Response(
                data, 
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

