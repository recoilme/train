package lib

import (
	"crypto/tls"
	"errors"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
)

func GetEditorialPhotos(key string, page int) (Photos, error) {
	client := http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	randomApi := "https://api.unsplash.com/photos"

	req, _ := http.NewRequest(http.MethodGet, randomApi, nil)

	req.Header.Add("Authorization", "Client-ID "+key)
	q := req.URL.Query()
	q.Add("per_page", "30")
	q.Add("page", strconv.Itoa(page))
	q.Add("order_by", "oldest")
	req.URL.RawQuery = q.Encode()

	log.Println(req.URL.String(), " , query: ", req.URL.Query())

	res, err := client.Do(req)

	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	log.Println("Response: ", res)

	if res.StatusCode == 200 {
		// Parse links
		body, _ := ioutil.ReadAll(res.Body)
		// log.Printf("%#v", string(body))
		photos, err := UnmarshalPhotos(body)
		log.Println("photos: ", len(photos), "err: ", err)

		if err != nil {
			return nil, err
		}

		return photos, nil
	} else {
		err = errors.New(res.Status)
		log.Println("Err: ", err)
		return nil, err
	}
}

// /collections/:id/photos
func GetLikedPhotos(key string, user string, page int) (Photos, error) {
	client := http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	likedApi := "https://api.unsplash.com/users/" + user + "/likes"

	req, _ := http.NewRequest(http.MethodGet, likedApi, nil)

	req.Header.Add("Authorization", "Client-ID "+key)
	q := req.URL.Query()
	q.Add("per_page", "30")
	q.Add("page", strconv.Itoa(page))
	q.Add("order_by", "oldest")
	req.URL.RawQuery = q.Encode()

	log.Println(req.URL.String(), " , query: ", req.URL.Query())

	res, err := client.Do(req)

	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	//log.Println("Response: ", res)

	if res.StatusCode == 200 {
		// Parse links
		body, _ := io.ReadAll(res.Body)
		//log.Printf("%#v", string(body))
		photos, err := UnmarshalPhotos(body)
		log.Println("photos: ", len(photos), "err: ", err)

		if err != nil {
			return nil, err
		}

		return photos, nil
	} else {
		err = errors.New(res.Status)
		log.Println("Err: ", err)
		return nil, err
	}
}

func GetTopicsPhotos(key string, topicId string, page int) (Photos, error) {
	client := http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	randomApi := "https://api.unsplash.com/topics/" + topicId + "/photos"

	req, _ := http.NewRequest(http.MethodGet, randomApi, nil)

	req.Header.Add("Authorization", "Client-ID "+key)
	q := req.URL.Query()
	q.Add("per_page", "30")
	q.Add("page", strconv.Itoa(page))
	q.Add("order_by", "oldest")
	req.URL.RawQuery = q.Encode()

	log.Println(req.URL.String(), " , query: ", req.URL.Query())

	res, err := client.Do(req)

	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	log.Println("Response: ", res)

	if res.StatusCode == 200 {
		// Parse links
		body, _ := ioutil.ReadAll(res.Body)
		// log.Printf("%#v", string(body))
		photos, err := UnmarshalPhotos(body)
		log.Println("photos: ", len(photos), "err: ", err)

		if err != nil {
			return nil, err
		}

		return photos, nil
	} else {
		err = errors.New(res.Status)
		log.Println("Err: ", err)
		return nil, err
	}
}

func GetRandomPhoto(key string) (Photos, error) {
	client := http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	randomApi := "https://api.unsplash.com/photos/random?count=30&client_id=" + AccessKey

	// req, err := http.NewRequest(http.MethodGet, randomApi, nil)

	// req.Header.Add("Authorization", "Client-ID "+key)
	// req.URL.Query().Add("count", "30")

	// log.Println(req.URL.String())

	// res, err := http.DefaultClient.Do(req)
	res, err := client.Get(randomApi)

	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	log.Println("Response: ", res)

	if res.StatusCode == 200 {
		// Parse links
		body, _ := ioutil.ReadAll(res.Body)
		// log.Printf("%#v", string(body))
		photos, err := UnmarshalPhotos(body)
		log.Println("photos: ", len(photos), "err: ", err)

		if err != nil {
			return nil, err
		}

		return photos, nil
		// for _, photo := range photos {
		// 	log.Println("id: ", *photo.ID, "url: ", *photo.Urls.Full)
		// 	DownloadFile(*photo.Urls.Full, *photo.ID)
		// }
	} else {
		err = errors.New(res.Status)
		log.Println("Err: ", err)
		return nil, err
	}
}

func DownloadFile(URL, fileName, description string) error {
	name := fileName + ".png"
	_, err := os.Stat("img/" + name)

	if err == nil {
		// File exist
		log.Println("Photo ", name, " already downloaded.")
		return nil
	}

	//downloadTokens <- struct{}{}
	client := http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	//Get the response bytes from the url
	response, err := client.Get(URL + "&w=1024&h=1024&fit=crop&crop=faces&fm=png")
	if err != nil {
		log.Println(fileName, " request error: ", err)
		return err
	}
	defer response.Body.Close()
	//<-downloadTokens

	if response.StatusCode != 200 {
		log.Println(fileName, " Received non 200 response code: ",
			response.StatusCode)
		return errors.New("received non 200 response code")
	}

	log.Println("Downloaded: ", fileName)
	//Create a empty file
	file, err := os.Create("img/" + name)
	if err != nil {
		log.Println("Fail create file ", fileName)
		return err
	}
	defer file.Close()

	//Write the bytes to the file
	_, err = io.Copy(file, response.Body)
	if err != nil {
		log.Println("Fail write file ", fileName)
		return err
	}

	if description == "" {
		return nil
	}
	desName := strings.Replace(name, ".png", ".txt", -1)
	fileDesc, err := os.Create("cpt/" + desName)
	if err != nil {
		log.Println("Fail create file ", fileName)
		return err
	}
	defer fileDesc.Close()

	_, err = io.Copy(fileDesc, strings.NewReader(description+","))
	if err != nil {
		log.Println("Fail write file desc ", fileName)
		return err
	}

	return nil
}
