<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Dataset</div>
      </v-card-title>
      <v-card-text>
        <template>
          <v-form v-model="valid" ref="form" lazy-validation>
            <v-text-field
              label="Title"
              v-model="new_dataset.title"
              required
            ></v-text-field>
            <v-text-field
              label="Description"
              v-model="new_dataset.description"
            ></v-text-field>
            <v-select
              label="File type"
              v-model="new_dataset.file_type"
              :items="file_types"
              deletable-chips
              chips
            ></v-select>
            <v-text-field
              label="Location"
              v-model="new_dataset.location"
            ></v-text-field>
            <v-slider
              label="Resolution X"
              v-model="new_dataset.resolution.x"
              min="1"
              max="5000"
            >
              <template v-slot:append>
                <v-text-field
                  v-model="new_dataset.resolution.x"
                  class="mt-0 pt-0"
                  type="number"
                  style="width: 60px"
                ></v-text-field>
              </template>
            </v-slider>
            <v-slider
              label="Resolution Y"
              v-model="new_dataset.resolution.y"
              min="1"
              max="5000"
            >
              <template v-slot:append>
                <v-text-field
                  v-model="new_dataset.resolution.y"
                  class="mt-0 pt-0"
                  type="number"
                  style="width: 60px"
                ></v-text-field>
              </template>
            </v-slider>
            <v-slider
              label="Resolution Z"
              v-model="new_dataset.resolution.z"
              min="1"
              max="5000"
            >
              <template v-slot:append>
                <v-text-field
                  v-model="new_dataset.resolution.z"
                  class="mt-0 pt-0"
                  type="number"
                  style="width: 60px"
                ></v-text-field>
              </template>
            </v-slider>
            <v-text-field
              label="Modality"
              v-model="new_dataset.modality"
              disabled
            ></v-text-field>
          </v-form>
        </template>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit" :disabled="!valid"> Save </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from "vue-property-decorator";
import { Dataset, DatasetUpdate, DatasetCreate, DatasetFileType } from "@/api";
import { defaultDataset } from "@/interfaces";
import {
  dispatchGetDatasets,
  dispatchCreateDataset,
} from "@/store/dataset/actions";

function filterUndefined(obj) {
  Object.keys(obj).forEach((key) => obj[key] === undefined && delete obj[key]);
  return obj;
}

@Component
export default class CreateDataset extends Vue {
  public new_dataset: DatasetCreate = defaultDataset();
  public file_types = Object.values(DatasetFileType);
  public valid = false;

  public async mounted() {
    await dispatchGetDatasets(this.$store);
    this.reset();
  }

  public reset() {
    this.new_dataset = defaultDataset();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      var filteredDataset: DatasetCreate = filterUndefined(this.new_dataset);
      console.log(filteredDataset);
      await dispatchCreateDataset(this.$store, filteredDataset);
      this.$router.push("/main/datasets");
    }
  }
}
</script>
