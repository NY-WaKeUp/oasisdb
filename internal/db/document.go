package db

import (
	"encoding/json"
	"fmt"
	"oasisdb/pkg/errors"
)

// Document represents a document
type Document struct {
	ID         string         `json:"id"`
	Vector     []float32      `json:"vector"`
	Parameters map[string]any `json:"parameters"`
	Dimension  int            `json:"dimension"`
}

type batchData struct {
	docKeys   [][]byte
	docValues [][]byte
	ids       []string
	vectors   [][]float32
}

// UpsertDocument inserts or updates a document
func (db *DB) UpsertDocument(collectionName string, doc *Document) error {
	// handle automatic embedding generation if requested
	if doc.Parameters != nil {
		if flag, ok := doc.Parameters["embedding"].(bool); ok && flag && len(doc.Vector) == 0 {
			text, okText := doc.Parameters["text"].(string)
			if !okText {
				return fmt.Errorf("text parameter is required for embedding when vector is not provided")
			}
			vec64, err := db.conf.EmbeddingProvider.Embed(text)
			if err != nil {
				return fmt.Errorf("failed to generate embedding: %w", err)
			}
			doc.Vector = float64SliceTo32(vec64)
			doc.Dimension = len(doc.Vector)
		}
	}

	// validate vector dimension
	if len(doc.Vector) != doc.Dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", doc.Dimension, len(doc.Vector))
	}

	// store document metadata
	docKey := fmt.Sprintf("doc:%s:%s", collectionName, doc.ID)
	docData, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if err := db.Storage.PutScalar([]byte(docKey), docData); err != nil {
		return err
	}

	// upsert vector index
	if err := db.IndexManager.AddVector(collectionName, doc.ID, doc.Vector); err != nil {
		return err
	}

	return nil
}

// GetDocument gets a document
func (db *DB) GetDocument(collectionName string, id string) (*Document, error) {
	docKey := fmt.Sprintf("doc:%s:%s", collectionName, id)
	data, exists, err := db.Storage.GetScalar([]byte(docKey))
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.ErrDocumentNotFound
	}

	var doc Document
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	return &doc, nil
}

// DeleteDocument deletes a document
func (db *DB) DeleteDocument(collectionName string, id string) error {
	docKey := fmt.Sprintf("doc:%s:%s", collectionName, id)
	if err := db.Storage.DeleteScalar([]byte(docKey)); err != nil {
		return err
	}

	if err := db.IndexManager.DeleteVector(collectionName, id); err != nil {
		return err
	}
	return nil
}

// SearchVectors returns top-k vector ids and distances
func (db *DB) SearchVectors(collectionName string, queryVector []float32, k int) ([]string, []float32, error) {
	index, err := db.IndexManager.GetIndex(collectionName)
	if err != nil {
		return nil, nil, err
	}
	searchResult, err := index.Search(queryVector, k)
	if err != nil {
		return nil, nil, err
	}
	return searchResult.IDs, searchResult.Distances, nil
}


// SearchDocuments returns top-k documents and distances
func (db *DB) SearchDocuments(collectionName string, queryDoc *Document, k int, filter map[string]interface{}) ([]*Document, []float32, error) {
    // Handle automatic embedding generation if requested
    if queryDoc.Parameters != nil {
        if flag, ok := queryDoc.Parameters["embedding"].(bool); ok && flag && len(queryDoc.Vector) == 0 {
            text, okText := queryDoc.Parameters["text"].(string)
            if !okText {
                return nil, nil, fmt.Errorf("text parameter is required for embedding when vector is not provided")
            }
            vec64, err := db.conf.EmbeddingProvider.Embed(text)
            if err != nil {
                return nil, nil, fmt.Errorf("failed to generate embedding: %w", err)
            }
            queryDoc.Vector = float64SliceTo32(vec64)
            queryDoc.Dimension = len(queryDoc.Vector)
        }
    }

    // Validate that query document has a vector
    if len(queryDoc.Vector) == 0 {
        return nil, nil, fmt.Errorf("query document must have a vector or embedding parameters")
    }

    // 1. get index
    index, err := db.IndexManager.GetIndex(collectionName)
    if err != nil {
        return nil, nil, err
    }

    // 2. search using hnsw index
    searchResult, err := index.Search(queryDoc.Vector, k)
    if err != nil {
        return nil, nil, err
    }

    // 3. get documents by ids
    docs := make([]*Document, len(searchResult.IDs))
    for i, id := range searchResult.IDs {
        doc, err := db.GetDocument(collectionName, id)
        if err != nil {
            return nil, nil, err
        }
        docs[i] = doc
    }

    // 4. return documents
    return docs, searchResult.Distances, nil
}

func (db *DB) prepareBatchData(collectionName string, docs []*Document) (*batchData, error) {
	// Get collection to validate dimension
	collection, err := db.GetCollection(collectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection: %w", err)
	}

	// Prepare batch data
	docKeys := make([][]byte, len(docs))
	docValues := make([][]byte, len(docs))
	ids := make([]string, len(docs))
	vectors := make([][]float32, len(docs))

	// Validate and prepare data
	for i, doc := range docs {
		// Automatic embedding generation for batch docs
		if doc.Parameters != nil {
			if flag, ok := doc.Parameters["embedding"].(bool); ok && flag && len(doc.Vector) == 0 {
				text, okText := doc.Parameters["text"].(string)
				if !okText {
					return nil, fmt.Errorf("text parameter is required for embedding when vector is not provided for document %s", doc.ID)
				}
				vec64, err := db.conf.EmbeddingProvider.Embed(text)
				if err != nil {
					return nil, fmt.Errorf("failed to generate embedding for document %s: %w", doc.ID, err)
				}
				doc.Vector = float64SliceTo32(vec64)
				doc.Dimension = len(doc.Vector)
			}
		}

		// Validate vector dimension
		if len(doc.Vector) != collection.Dimension {
			return nil, fmt.Errorf("vector dimension mismatch for document %s: expected %d, got %d",
				doc.ID, collection.Dimension, len(doc.Vector))
		}
		doc.Dimension = collection.Dimension

		// Prepare document key and value
		docKey := fmt.Sprintf("doc:%s:%s", collectionName, doc.ID)
		docData, err := json.Marshal(doc)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal document %s: %w", doc.ID, err)
		}

		docKeys[i] = []byte(docKey)
		docValues[i] = docData
		ids[i] = doc.ID
		vectors[i] = doc.Vector
	}

	return &batchData{
		docKeys:   docKeys,
		docValues: docValues,
		ids:       ids,
		vectors:   vectors,
	}, nil
}

func (db *DB) BuildIndex(collectionName string, docs []*Document) error {
	// Prepare batch data
	batchData, err := db.prepareBatchData(collectionName, docs)
	if err != nil {
		return err
	}

	// Batch store document metadata
	if err := db.Storage.BatchPutScalar(batchData.docKeys, batchData.docValues); err != nil {
		return fmt.Errorf("failed to batch store documents: %w", err)
	}

	// Build index
	if err := db.IndexManager.BuildIndex(collectionName, batchData.ids, batchData.vectors); err != nil {
		return fmt.Errorf("failed to batch update vector index: %w", err)
	}

	return nil
}

func (db *DB) BatchUpsertDocuments(collectionName string, docs []*Document) error {
	// Prepare batch data
	batchData, err := db.prepareBatchData(collectionName, docs)
	if err != nil {
		return err
	}

	// Batch store document metadata
	if err := db.Storage.BatchPutScalar(batchData.docKeys, batchData.docValues); err != nil {
		return fmt.Errorf("failed to batch store documents: %w", err)
	}

	// Batch update vector index
	if err := db.IndexManager.AddVectorBatch(collectionName, batchData.ids, batchData.vectors); err != nil {
		return fmt.Errorf("failed to batch update vector index: %w", err)
	}

	return nil
}

// float64SliceTo32 converts a slice of float64 to float32
func float64SliceTo32(src []float64) []float32 {
	res := make([]float32, len(src))
	for i, v := range src {
		res[i] = float32(v)
	}
	return res
}
