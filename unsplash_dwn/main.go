package main

import (
	"flag"
	"log"
	"math/rand"
	"time"
	"unsplash_dwn/pkg/api"
)

var AccessKey string
var UserName = "babakasotona"

//var wg sync.WaitGroup

//var downloadTokens = make(chan struct{}, 10)

var topics Topics

type Topics struct {
	TopicsIndx uint64
	TopicsId   string
	PageOffset uint64
}

var topicsKeys = []string{}

func main() {
	flag.StringVar(&AccessKey, "c", "", "Client Access Key.")
	flag.Parse()

	if AccessKey == "" {
		log.Panicln("Missing access key.")
	}

	for {

		current := time.Now()
		var photos api.Photos

		// Crawl Topics photos
		photos, err := api.GetLikedPhotos(AccessKey,
			UserName,
			int(topics.PageOffset))

		if err != nil {
			time.Sleep(30 * time.Second)
			continue
		}

		if len(photos) == 0 {
			// Next topics
			topics.PageOffset = 1
			topics.TopicsIndx += 1
			// avoid out of range index.
			topics.TopicsId = topicsKeys[topics.TopicsIndx]

			log.Println("Topics ", topics, ", next page: ", topics.PageOffset)
			time.Sleep(72 * time.Second)
			continue
		}
		hasErr := false
		for _, photo := range photos {

			err := api.DownloadFile(photo.Urls.Raw, photo.ID, photo.AltDescription)
			if !hasErr && err != nil {
				hasErr = true
			}
		}

		elapse := int32(time.Since(current).Seconds())

		log.Println("Time used: ", elapse, " seconds.")

		// Next page query only for all downloads success.
		if !hasErr {
			topics.PageOffset += 1
			log.Println("Topics ", topics, ", next page: ", topics.PageOffset)
		}

		diff := 72 - elapse
		offset := int32(0)
		if diff <= 0 {
			offset = 72
		} else if diff > 72 {
			offset = 72
		} else {
			offset = 0
		}
		delay := offset + rand.Int31n(10)
		log.Println("Sleep: ", delay, " seconds.")
		time.Sleep(time.Duration(delay) * time.Second)
	}
}
