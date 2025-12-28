from rest_framework import serializers
from .models import Document, Keyword, DocumentKeyword

class KeywordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Keyword
        fields = ['id', 'keyword']

class DocumentKeywordSerializer(serializers.ModelSerializer):
    keyword = serializers.CharField(source='keyword.keyword')
    
    class Meta:
        model = DocumentKeyword
        fields = ['keyword', 'relevance_score']

class DocumentSerializer(serializers.ModelSerializer):
    keywords = DocumentKeywordSerializer(many=True, read_only=True)
    keyword_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id', 
            'filename', 
            'file', 
            'file_type', 
            'upload_date', 
            'processed', 
            'summary',  # NEW FIELD
            'keywords',
            'keyword_count',
        ]
    
    def get_keyword_count(self, obj):
        return obj.keywords.count()

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    
    def validate_file(self, value):
        # Validate file size (max 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 10MB")
        
        # Validate file extension
        file_extension = value.name.split('.')[-1].lower()
        if file_extension not in ['pdf', 'docx', 'doc', 'txt']:
            raise serializers.ValidationError(
                "Only PDF, DOCX, and TXT files are supported"
            )
        
        return value
