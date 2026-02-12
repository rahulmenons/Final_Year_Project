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
    evaluate_and_save,
    RFPMetadataExtractor,
    index_document,
)


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)

    @action(detail=False, methods=["post"], url_path="upload")
    def upload_document(self, request):

        serializer = FileUploadSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = serializer.validated_data["file"]
        file_extension = uploaded_file.name.split(".")[-1].lower()

        try:
            # 1 Parse document
            parser = DocumentParser()
            text = parser.parse(uploaded_file, file_extension)

            if not text:
                return Response({"error": "Could not extract text"}, status=400)

            # 2 Keywords
            extractor = KeywordExtractor()
            keywords_with_scores = extractor.extract_keywords(text, top_n=15)

            # 3 Summary
            summarizer = DocumentSummarizer()
            summary_text = summarizer.generate_summary(text, max_length=350)

            # 4 RAG Metadata
            metadata_extractor = RFPMetadataExtractor()
            meta = metadata_extractor.extract_metadata()

            rfp_budget = meta.get("budget_in_inr") or 0
            rfp_timeline_weeks = meta.get("timeline_weeks") or 0

            # 5 Save
            with transaction.atomic():

                document = Document.objects.create(
                    filename=uploaded_file.name,
                    file=uploaded_file,
                    file_type=file_extension,
                    content_preview=text[:1000],
                    summary=summary_text,
                    processed=False,
                    rfp_budget=rfp_budget,
                    rfp_timeline_weeks=rfp_timeline_weeks,

                )

                # RAG indexing
                index_document(document, text)

                for keyword_text, score in keywords_with_scores:
                    keyword, _ = Keyword.objects.get_or_create(keyword=keyword_text)
                    DocumentKeyword.objects.create(
                        document=document,
                        keyword=keyword,
                        relevance_score=float(score),
                    )

                evaluation = evaluate_and_save(document)

            response_serializer = DocumentSerializer(document)
            data = response_serializer.data

            # ✅ FIXED evaluation mapping
            data["evaluation"] = {
                "technical_fit_score": evaluation.technical_fit_score,
                "budget_fit_score": evaluation.budget_fit_score,
                "timeline_fit_score": evaluation.timeline_fit_score,
                "capacity_fit_score": evaluation.capacity_fit_score,
                "overall_fit_score": evaluation.overall_fit_score,
                "decision": evaluation.decision,
                "reasoning": evaluation.reasoning,
            }

            # ✅ FIXED metadata mapping
            data["rfp_metadata"] = {
                "budget_in_inr": rfp_budget,
                "timeline_weeks": rfp_timeline_weeks,

                "extraction_confidence": meta.get("confidence"),
                "extraction_notes": meta.get("notes"),
            }

            return Response(data, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(e)
            return Response({"error": str(e)}, status=500)
