// Copyright 2026 The Neo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed books/*
var Books embed.FS

// Book is a book
type Book struct {
	Name string
	Text []byte
}

// LoadBooks loads books
func LoadBooks() []Book {
	books := []Book{
		{Name: "10.txt.utf-8.bz2"},
		//{Name: "11.txt.utf-8.bz2"},
		//{Name: "43.txt.utf-8.bz2"},
		{Name: "pg74.txt.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		//{Name: "145.txt.utf-8.bz2"},
		//{Name: "768.txt.utf-8.bz2"},
		//{Name: "1260.txt.utf-8.bz2"},
		//{Name: "1342.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		//{Name: "2641.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
		//{Name: "37106.txt.utf-8.bz2"},
		//{Name: "64317.txt.utf-8.bz2"},
		//{Name: "67979.txt.utf-8.bz2"},
	}
	load := func(book string) []byte {
		file, err := Books.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		return data
	}
	for i := range books {
		books[i].Text = load(fmt.Sprintf("books/%s", books[i].Name))
	}
	return books
}

// Euclidean computes the euclidean distance between all row vectors and all row vectors
func EuclideanReal(k tf64.Continuation, node int, a, b *tf64.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sizeA, sizeB := tf64.NewV(a.S[1], b.S[1]), len(a.X), len(b.X)
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, sum := a.X[i:i+width], b.X[ii:ii+width], 0.0
			for j, ax := range av {
				diff := (ax - bv[j])
				sum += diff * diff
			}
			c.X = append(c.X, math.Sqrt(sum))
		}
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, cx, ad, bd, d := a.X[i:i+width], b.X[ii:ii+width], c.X[index], a.D[i:i+width], b.D[ii:ii+width], c.D[index]
			for j, ax := range av {
				if cx == 0 {
					continue
				}
				ad[j] += (ax - bv[j]) * d / cx
				bd[j] += (bv[j] - ax) * d / cx
			}
			index++
		}
	}
	return false
}

// QR is a real model
type QR struct {
	Iteration int
	Rng       *rand.Rand
	Set       *tf64.Set
}

// NewQG creates a new real model
func NewQR(rows, cols int) QR {
	rng := rand.New(rand.NewSource(1))

	set := tf64.NewSet()
	set.Add("x", 2, rows)
	set.Add("y", 2, rows)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	return QR{
		Rng: rng,
		Set: &set,
	}
}

// Iterate iterates the g model
func (q *QR) Iterate(iterations int) {
	const Eta = 1e-1
	drop := .3
	dropout := map[string]interface{}{
		"rng":  q.Rng,
		"drop": &drop,
	}

	euclidean := tf64.B(EuclideanReal)

	l0 := tf64.Mul(tf64.Dropout(tf64.Square(q.Set.Get("y")), dropout),
		tf64.Inv(euclidean(q.Set.Get("x"), q.Set.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(tf64.Mul(tf64.Dropout(tf64.Square(q.Set.Get("x")), dropout),
		tf64.Inv(euclidean(q.Set.Get("y"), q.Set.Get("y")))), l0))

	var l float64
	for range iterations {
		iteration := q.Iteration
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		q.Set.Zero()
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(l) || math.IsInf(l, 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range q.Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range q.Set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		q.Iteration++
	}
	fmt.Println(l)

	x := q.Set.ByName["x"]
	minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
	for i := range x.S[1] {
		x, y := x.X[i*x.S[0]], x.X[i*x.S[0]+1]
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}
	halfX, halfY := (maxX-minX)/2, (maxY-minY)/2
	var counts [4]int
	for i := range x.S[1] {
		x, y := x.X[i*x.S[0]], x.X[i*x.S[0]+1]
		if x < minX+halfX && y < minY+halfY {
			counts[0]++
		} else if x > minX+halfX && x < maxX && y < minY+halfY {
			counts[1]++
		} else if x < minX+halfX && y > minY+halfY && y < maxY {
			counts[2]++
		} else {
			counts[3]++
		}
	}

	y := q.Set.ByName["y"]
	minX, maxX, minY, maxY = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
	for i := range y.S[1] {
		x, y := y.X[i*y.S[0]], y.X[i*y.S[0]+1]
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}
	halfX, halfY = (maxX-minX)/2, (maxY-minY)/2
	counts = [4]int{}
	for i := range y.S[1] {
		x, y := y.X[i*y.S[0]], y.X[i*y.S[0]+1]
		if x < minX+halfX && y < minY+halfY {
			counts[0]++
		} else if x > minX+halfX && x < maxX && y < minY+halfY {
			counts[1]++
		} else if x < minX+halfX && y > minY+halfY && y < maxY {
			counts[2]++
		} else {
			counts[3]++
		}
	}
}

func main() {
	LoadBooks()
}
