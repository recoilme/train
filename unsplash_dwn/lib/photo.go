// This file was generated from JSON Schema using quicktype, do not modify it directly.
// To parse and unparse this JSON data, add this code to your project and do:
//
//    photos, err := UnmarshalPhotos(bytes)
//    bytes, err = photos.Marshal()

package lib

import "encoding/json"

type Photos []Photo

func UnmarshalPhotos(data []byte) (Photos, error) {
	var r Photos
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *Photos) Marshal() ([]byte, error) {
	return json.Marshal(r)
}

type Photo struct {
	ID             string     `json:"id,omitempty"`
	AltDescription string     `json:"alt_description,omitempty"`
	Urls           Urls       `json:"urls,omitempty"`
	Links          PhotoLinks `json:"links,omitempty"`
	Exif           Exif       `json:"exif,omitempty"`
	Tags           []struct {
		Title string `json:"title"`
	} `json:"tags,omitempty"`
}

type Exif struct {
	Make         string `json:"make"`
	Model        string `json:"model"`
	Name         string `json:"name"`
	ExposureTime string `json:"exposure_time"`
	Aperture     string `json:"aperture"`
	FocalLength  string `json:"focal_length"`
	ISO          int64  `json:"iso"`
}

type PhotoLinks struct {
	Self             string `json:"self,omitempty"`
	HTML             string `json:"html,omitempty"`
	Download         string `json:"download,omitempty"`
	DownloadLocation string `json:"download_location,omitempty"`
}

type Urls struct {
	Raw     string `json:"raw,omitempty"`
	Full    string `json:"full,omitempty"`
	Regular string `json:"regular,omitempty"`
	Small   string `json:"small,omitempty"`
	Thumb   string `json:"thumb,omitempty"`
	SmallS3 string `json:"small_s3,omitempty"`
}
